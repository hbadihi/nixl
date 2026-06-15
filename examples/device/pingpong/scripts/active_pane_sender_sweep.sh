#!/usr/bin/env bash
set -euo pipefail

# Run this inside the already-attached sender container pane after starting
# active_pane_receiver_sweep.sh in the receiver pane.

REPO_ROOT="${REPO_ROOT:-/workspace/external/nixl-cpu-proxy}"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/build/examples/device/pingpong}"

GPU="${SENDER_GPU:-0}"

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
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-300}"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/profile_results/active-pane-current.env}"
SENDER_STATE_FILE="${SENDER_STATE_FILE:-$REPO_ROOT/profile_results/active-pane-sender.env}"

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

infer_rdma_dev_for_gpu() {
  if [[ -n "${SENDER_RDMA_DEV:-${RDMA_DEV:-}}" ]]; then
    echo "${SENDER_RDMA_DEV:-$RDMA_DEV}"
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

  echo "ERROR: could not infer RDMA device; set SENDER_RDMA_DEV or RDMA_DEV" >&2
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
  echo "ERROR: could not infer IPv4 address for netdev $netdev; set SENDER_IP" >&2
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

if [[ -z "${OUT_DIR:-}" ]]; then
  echo "waiting for receiver state: $STATE_FILE"
  deadline=$((SECONDS + WAIT_TIMEOUT_S))
  while true; do
    if [[ -f "$STATE_FILE" ]]; then
      state_out_dir="$(grep "^OUT_DIR=" "$STATE_FILE" | cut -d= -f2- | tr -d "'" || true)"
      if [[ -n "$state_out_dir" && ! -e "$state_out_dir/.coord/receiver.done" && ! -e "$state_out_dir/receiver.done" ]]; then
        # shellcheck disable=SC1090
        source "$STATE_FILE"
        break
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: timed out waiting for active receiver state file: $STATE_FILE" >&2
      exit 2
    fi
    sleep 1
  done
fi
mkdir -p "$OUT_DIR"
COORD_DIR="${COORD_DIR:-$OUT_DIR/.coord}"
mkdir -p "$COORD_DIR"
RUNS_CSV="$OUT_DIR/runs.csv"
init_runs_csv

RDMA_DEV="$(infer_rdma_dev_for_gpu)"
NETDEV="$(netdev_for_rdma_dev "$RDMA_DEV")"
INFERRED_SENDER_IP="$(ipv4_for_netdev "$NETDEV")"
SENDER_IP="${SENDER_IP:-$INFERRED_SENDER_IP}"
UCX_NET_DEVICES="${SENDER_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-$RDMA_DEV:1,cuda${GPU}-${RDMA_DEV}:1}}"
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

cat >"$SENDER_STATE_FILE.tmp" <<EOF
SENDER_IP='$SENDER_IP'
SENDER_RDMA_DEV='$RDMA_DEV'
SENDER_NETDEV='$NETDEV'
SENDER_UCX_NET_DEVICES='$UCX_NET_DEVICES'
EOF
mv "$SENDER_STATE_FILE.tmp" "$SENDER_STATE_FILE"

echo "sender sweep"
echo "  out_dir=$OUT_DIR"
echo "  state_file=$STATE_FILE"
echo "  local sender_ip=$SENDER_IP rdma=$RDMA_DEV netdev=$NETDEV"
echo "  peer receiver_ip=$RECEIVER_IP rdma=${RECEIVER_RDMA_DEV:-unknown} netdev=${RECEIVER_NETDEV:-unknown}"
echo "  ops=$OPS levels=$LEVELS measure_modes=$MEASURE_MODES proxy_stats_modes=$PROXY_STATS_MODES"
echo "  sizes=$SIZES iters=$ITERS warmup=$WARMUP base_port=$BASE_PORT"
echo "  UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=${UCX_TLS:-<unset>}"
echo "  wait_timeout_s=$WAIT_TIMEOUT_S"

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
          echo "WAIT tag=$tag ready=$ready_file"
          deadline=$((SECONDS + WAIT_TIMEOUT_S))
          while [[ ! -f "$ready_file" ]]; do
            if (( SECONDS >= deadline )); then
              echo "ERROR: timed out waiting for receiver ready file: $ready_file" >&2
              exit 2
            fi
            sleep 0.2
          done

          echo "SEND tag=$tag listen=$((port + 1)) peer_port=$port"
          args=(
            --role sender
            --peer-ip "$RECEIVER_IP"
            --peer-port "$port"
            --listen-port "$((port + 1))"
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
              NIXL_PROXY_STATS=0 "$bin" "${args[@]}" >"$OUT_DIR/$sender_out" 2>"$OUT_DIR/$sender_err" || exit_status=$?
            else
              NIXL_PROXY_STATS=1 "$bin" "${args[@]}" >"$OUT_DIR/$sender_out" 2>"$OUT_DIR/$sender_err" || exit_status=$?
            fi
          else
            "$bin" "${args[@]}" >"$OUT_DIR/$sender_out" 2>"$OUT_DIR/$sender_err" || exit_status=$?
          fi

          append_run_index sender "$op" "$level" "$measure" "$stats_tag" "$variant" "$size" "$((port + 1))" "$port" "$sender_out" "$sender_err" "$receiver_out" "$receiver_err" "$exit_status"
          if [[ "$exit_status" != "0" ]]; then
            echo "ERROR: sender run failed for tag=$tag exit_status=$exit_status" >&2
            exit "$exit_status"
          fi

          BASE_PORT="$((BASE_PORT + 2))"
          sleep 1
        done
      done
      done
    done
  done
done

echo "sender sweep complete: $OUT_DIR"
echo
echo "summary:"
grep -h '^op=' "$OUT_DIR"/send_*.out || true
echo
echo "runs csv: $RUNS_CSV"
