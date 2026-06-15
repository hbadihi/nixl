#!/usr/bin/env bash
set -euo pipefail

# Run this inside the already-attached receiver container pane.

REPO_ROOT="${REPO_ROOT:-/workspace/external/nixl-cpu-proxy}"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/build/examples/device/pingpong}"

GPU="${RECEIVER_GPU:-0}"

SIZES="${SIZES:-8 64 512 4096 32768 262144 1048576}"
# atomicAdd transfers a fixed 8-byte payload, so sweeping sizes is meaningless for it.
# We sweep SIZES for `put` ops and run a single fixed-size pass for `atomic-flag`.
ATOMIC_FLAG_SIZE="${ATOMIC_FLAG_SIZE:-8}"
OPS="${OPS:-put atomic-flag}"
LEVELS="${LEVELS:-thread warp}"
MEASURE_MODES="${MEASURE_MODES:-off}"
PROXY_STATS_MODES="${PROXY_STATS_MODES:-on off}"
ITERS="${ITERS:-5000}"
WARMUP="${WARMUP:-500}"
BASE_PORT="${BASE_PORT:-21000}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/profile_results/active-pane-$RUN_ID}"
mkdir -p "$OUT_DIR"
COORD_DIR="${COORD_DIR:-$OUT_DIR/.coord}"
mkdir -p "$COORD_DIR"
ln -sfn "$OUT_DIR" "$REPO_ROOT/profile_results/active-pane-latest"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/profile_results/active-pane-current.env}"
SENDER_STATE_FILE="${SENDER_STATE_FILE:-$REPO_ROOT/profile_results/active-pane-sender.env}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-300}"

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

infer_rdma_dev_for_gpu() {
  if [[ -n "${RECEIVER_RDMA_DEV:-${RDMA_DEV:-}}" ]]; then
    echo "${RECEIVER_RDMA_DEV:-$RDMA_DEV}"
    return
  fi

  local topo gpu_label header row best_nic relation nic_name
  topo="$(nvidia-smi topo -m 2>/dev/null | strip_ansi || true)"
  gpu_label="GPU$GPU"
  header="$(printf '%s\n' "$topo" | awk '/GPU0/ && /NIC0/ {print; exit}')"
  row="$(printf '%s\n' "$topo" | awk -v gpu="$gpu_label" '$1 == gpu {print; exit}')"

  if [[ -n "$header" && -n "$row" ]]; then
    read -r -a header_cols <<<"$header"
    read -r -a row_cols <<<"$row"
    for wanted in PIX PXB PHB NODE SYS; do
      for i in "${!header_cols[@]}"; do
        [[ "${header_cols[$i]}" == NIC* ]] || continue
        relation="${row_cols[$((i + 1))]:-}"
        if [[ "$relation" == "$wanted" ]]; then
          best_nic="${header_cols[$i]}"
          nic_name="$(printf '%s\n' "$topo" | awk -v nic="$best_nic:" '$1 == nic {print $2; exit}')"
          if [[ -n "$nic_name" ]]; then
            echo "$nic_name"
            return
          fi
        fi
      done
    done
  fi

  for dev in /sys/class/infiniband/mlx5_*; do
    [[ -e "$dev" ]] || continue
    basename "$dev"
    return
  done

  echo "ERROR: could not infer RDMA device; set RECEIVER_RDMA_DEV or RDMA_DEV" >&2
  exit 2
}

netdev_for_rdma_dev() {
  local rdma_dev="$1"
  local net_dir="/sys/class/infiniband/$rdma_dev/device/net"
  if [[ -d "$net_dir" ]]; then
    for netdev in "$net_dir"/*; do
      [[ -e "$netdev" ]] || continue
      basename "$netdev"
      return
    done
  fi
  echo "ERROR: could not map RDMA device $rdma_dev to netdev" >&2
  exit 2
}

ipv4_for_netdev() {
  local netdev="$1"
  local ip_addr
  ip_addr="$(ip -o -4 addr show dev "$netdev" scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"
  if [[ -n "$ip_addr" ]]; then
    echo "$ip_addr"
    return
  fi
  echo "ERROR: could not infer IPv4 address for netdev $netdev; set RECEIVER_IP" >&2
  exit 2
}

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

measure_args_for_mode() {
  local mode="$1"
  case "$mode" in
    on|measure|measure-submit|true|1)
      printf '%s\n' "--measure-submit"
      ;;
    off|no-measure|no-measure-submit|false|0)
      printf '%s\n' "--no-measure-submit"
      ;;
    *)
      echo "ERROR: unknown measure mode '$mode'; expected on/off or measure-submit/no-measure-submit" >&2
      exit 2
      ;;
  esac
}

init_runs_csv() {
  : "${RUNS_LOCK:=$RUNS_CSV.lock}"
  (
    flock -x 9
    if [[ ! -s "$RUNS_CSV" ]]; then
      echo "role,op,level,measure,proxy_stats,variant,msg_size,iters,warmup,listen_port,peer_port,sender_out,sender_err,receiver_out,receiver_err,exit_status" >"$RUNS_CSV"
    fi
  ) 9>"$RUNS_LOCK"
}

append_run_index() {
  local role="$1" op="$2" level="$3" measure="$4" stats_tag="$5" variant="$6" size="$7"
  local listen_port="$8" peer_port="$9" sender_out="${10}" sender_err="${11}" receiver_out="${12}" receiver_err="${13}" exit_status="${14}"
  local line
  line="$(csv_escape "$role"),$(csv_escape "$op"),$(csv_escape "$level"),$(csv_escape "$measure"),$(csv_escape "$stats_tag"),$(csv_escape "$variant"),$(csv_escape "$size"),$(csv_escape "$ITERS"),$(csv_escape "$WARMUP"),$(csv_escape "$listen_port"),$(csv_escape "$peer_port"),$(csv_escape "$sender_out"),$(csv_escape "$sender_err"),$(csv_escape "$receiver_out"),$(csv_escape "$receiver_err"),$(csv_escape "$exit_status")"
  : "${RUNS_LOCK:=$RUNS_CSV.lock}"
  (
    flock -x 9
    printf '%s\n' "$line" >>"$RUNS_CSV"
  ) 9>"$RUNS_LOCK"
}

write_run_manifest() {
  RUN_MANIFEST_PATH="$OUT_DIR/run_manifest.json" \
  RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" REPO_ROOT="$REPO_ROOT" BIN_DIR="$BIN_DIR" \
  SIZES="$SIZES" OPS="$OPS" LEVELS="$LEVELS" MEASURE_MODES="$MEASURE_MODES" \
  PROXY_STATS_MODES="$PROXY_STATS_MODES" ITERS="$ITERS" WARMUP="$WARMUP" \
  BASE_PORT="$BASE_PORT" WAIT_TIMEOUT_S="$WAIT_TIMEOUT_S" \
  RECEIVER_GPU="$GPU" RECEIVER_IP="$RECEIVER_IP" RECEIVER_RDMA_DEV="$RDMA_DEV" \
  RECEIVER_NETDEV="$NETDEV" RECEIVER_UCX_NET_DEVICES="$UCX_NET_DEVICES" \
  SENDER_IP="${SENDER_IP:-}" SENDER_RDMA_DEV="${SENDER_RDMA_DEV:-}" \
  SENDER_NETDEV="${SENDER_NETDEV:-}" SENDER_UCX_NET_DEVICES="${SENDER_UCX_NET_DEVICES:-}" \
  RECEIVER_SCRIPT_PATH="$(realpath "$0" 2>/dev/null || printf '%s' "$0")" \
  SENDER_SCRIPT_PATH="$REPO_ROOT/examples/device/pingpong/scripts/active_pane_sender_sweep.sh" \
  GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

def words(name):
    return os.environ.get(name, "").split()

manifest = {
    "run_id": os.environ["RUN_ID"],
    "out_dir": os.environ["OUT_DIR"],
    "repo_root": os.environ["REPO_ROOT"],
    "bin_dir": os.environ["BIN_DIR"],
    "matrix": {
        "sizes": words("SIZES"),
        "ops": words("OPS"),
        "levels": words("LEVELS"),
        "measure_modes": words("MEASURE_MODES"),
        "proxy_stats_modes": words("PROXY_STATS_MODES"),
    },
    "iterations": {
        "iters": int(os.environ["ITERS"]),
        "warmup": int(os.environ["WARMUP"]),
        "base_port_start": int(os.environ["BASE_PORT"]),
        "wait_timeout_s": int(os.environ["WAIT_TIMEOUT_S"]),
    },
    "receiver": {
        "gpu": os.environ["RECEIVER_GPU"],
        "ip": os.environ["RECEIVER_IP"],
        "rdma_dev": os.environ["RECEIVER_RDMA_DEV"],
        "netdev": os.environ["RECEIVER_NETDEV"],
        "ucx_net_devices": os.environ["RECEIVER_UCX_NET_DEVICES"],
    },
    "sender": {
        "ip": os.environ.get("SENDER_IP", ""),
        "rdma_dev": os.environ.get("SENDER_RDMA_DEV", ""),
        "netdev": os.environ.get("SENDER_NETDEV", ""),
        "ucx_net_devices": os.environ.get("SENDER_UCX_NET_DEVICES", ""),
    },
    "ucx": {
        "ucx_tls": os.environ.get("UCX_TLS", ""),
        "ucx_max_rma_rails": os.environ.get("UCX_MAX_RMA_RAILS", ""),
        "nixl_log_level": os.environ.get("NIXL_LOG_LEVEL", ""),
    },
    "scripts": {
        "receiver": os.environ["RECEIVER_SCRIPT_PATH"],
        "sender": os.environ["SENDER_SCRIPT_PATH"],
        "git_rev": os.environ["GIT_REV"],
    },
}
Path(os.environ["RUN_MANIFEST_PATH"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

RDMA_DEV="$(infer_rdma_dev_for_gpu)"
NETDEV="$(netdev_for_rdma_dev "$RDMA_DEV")"
INFERRED_RECEIVER_IP="$(ipv4_for_netdev "$NETDEV")"
RECEIVER_IP="${RECEIVER_IP:-$INFERRED_RECEIVER_IP}"
UCX_NET_DEVICES="${RECEIVER_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-$RDMA_DEV:1,cuda${GPU}-${RDMA_DEV}:1}}"
export UCX_NET_DEVICES
export UCX_MAX_RMA_RAILS="${UCX_MAX_RMA_RAILS:-1}"
export NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-FATAL}"

# Do not force UCX_TLS by default. Direct GPU Device API needs UCX to select
# rc_gda plus whatever auxiliary/control lane it requires. UCX_NET_DEVICES
# includes the matching CUDA/NIC device lane so handle packing can use rc_gda.
# To experiment, set ACTIVE_PANE_UCX_TLS explicitly before running this script.
if [[ -n "${ACTIVE_PANE_UCX_TLS:-}" ]]; then
  export UCX_TLS="$ACTIVE_PANE_UCX_TLS"
else
  unset UCX_TLS
fi

cd "$REPO_ROOT"

RUNS_CSV="$OUT_DIR/runs.csv"
init_runs_csv
rm -f "$SENDER_STATE_FILE" "$COORD_DIR/receiver.done" "$OUT_DIR/receiver.done"
cat >"$STATE_FILE.tmp" <<EOF
OUT_DIR='$OUT_DIR'
COORD_DIR='$COORD_DIR'
SIZES='$SIZES'
OPS='$OPS'
LEVELS='$LEVELS'
MEASURE_MODES='$MEASURE_MODES'
PROXY_STATS_MODES='$PROXY_STATS_MODES'
ITERS='$ITERS'
WARMUP='$WARMUP'
BASE_PORT='$BASE_PORT'
RECEIVER_IP='$RECEIVER_IP'
RECEIVER_RDMA_DEV='$RDMA_DEV'
RECEIVER_NETDEV='$NETDEV'
RECEIVER_UCX_NET_DEVICES='$UCX_NET_DEVICES'
EOF
mv "$STATE_FILE.tmp" "$STATE_FILE"

echo "waiting for sender state: $SENDER_STATE_FILE"
deadline=$((SECONDS + WAIT_TIMEOUT_S))
while [[ ! -f "$SENDER_STATE_FILE" ]]; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for sender state file: $SENDER_STATE_FILE" >&2
    exit 2
  fi
  sleep 1
done
# shellcheck disable=SC1090
source "$SENDER_STATE_FILE"
write_run_manifest

echo "receiver sweep"
echo "  out_dir=$OUT_DIR"
echo "  state_file=$STATE_FILE"
echo "  local receiver_ip=$RECEIVER_IP rdma=$RDMA_DEV netdev=$NETDEV"
echo "  peer sender_ip=$SENDER_IP rdma=${SENDER_RDMA_DEV:-unknown} netdev=${SENDER_NETDEV:-unknown}"
echo "  ops=$OPS levels=$LEVELS measure_modes=$MEASURE_MODES proxy_stats_modes=$PROXY_STATS_MODES"
echo "  sizes=$SIZES iters=$ITERS warmup=$WARMUP base_port=$BASE_PORT"
echo "  UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=${UCX_TLS:-<unset>}"

for op in $OPS; do
  if [[ "$op" == "atomic-flag" ]]; then
    op_sizes="$ATOMIC_FLAG_SIZE"
  else
    op_sizes="$SIZES"
  fi
  for level in $LEVELS; do
    for measure in $MEASURE_MODES; do
      for stats in $PROXY_STATS_MODES; do
        for size in $op_sizes; do
          for variant in direct proxy; do
          if [[ "$variant" == "direct" && "$stats" != "${PROXY_STATS_MODES%% *}" ]]; then
            continue
          fi
          if [[ "$variant" == "direct" ]]; then
            bin="$BIN_DIR/nixl_device_pingpong"
            stats_tag="na"
          else
            bin="$BIN_DIR/nixl_device_pingpong_proxy"
            stats_tag="$stats"
          fi

          tag="${op}_${level}_${measure}_stats-${stats_tag}_${variant}_${size}"
          port="$BASE_PORT"
          ready_file="$COORD_DIR/ready_${tag}_${port}"
          done_file="$COORD_DIR/done_${tag}_${port}"
          rm -f "$ready_file" "$done_file"
          echo "RECV tag=$tag listen=$port peer_port=$((port + 1))"
          echo "port=$port" >"$ready_file"

          args=(
            --role receiver
            --peer-ip "$SENDER_IP"
            --peer-port "$((port + 1))"
            --listen-port "$port"
            --msg-size "$size"
            --iters "$ITERS"
            --warmup "$WARMUP"
            --gpu "$GPU"
            --op "$op"
          )
          if [[ "$level" == "warp" ]]; then
            args+=(--warp)
          fi
          args+=("$(measure_args_for_mode "$measure")")

          sender_out="send_${tag}.out"
          sender_err="send_${tag}.err"
          receiver_out="recv_${tag}.out"
          receiver_err="recv_${tag}.err"
          exit_status=0

          if [[ "$variant" == "proxy" ]]; then
            if [[ "$stats" == "off" || "$stats" == "0" || "$stats" == "false" ]]; then
              NIXL_PROXY_STATS=0 "$bin" "${args[@]}" >"$OUT_DIR/$receiver_out" 2>"$OUT_DIR/$receiver_err" || exit_status=$?
            else
              NIXL_PROXY_STATS=1 "$bin" "${args[@]}" >"$OUT_DIR/$receiver_out" 2>"$OUT_DIR/$receiver_err" || exit_status=$?
            fi
          else
            "$bin" "${args[@]}" >"$OUT_DIR/$receiver_out" 2>"$OUT_DIR/$receiver_err" || exit_status=$?
          fi
          append_run_index receiver "$op" "$level" "$measure" "$stats_tag" "$variant" "$size" "$port" "$((port + 1))" "$sender_out" "$sender_err" "$receiver_out" "$receiver_err" "$exit_status"
          touch "$done_file"
          if [[ "$exit_status" != "0" ]]; then
            echo "ERROR: receiver run failed for tag=$tag exit_status=$exit_status" >&2
            exit "$exit_status"
          fi

          BASE_PORT="$((BASE_PORT + 2))"
        done
      done
      done
    done
  done
done

touch "$COORD_DIR/receiver.done"
echo "receiver sweep complete: $OUT_DIR"
