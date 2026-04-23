#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# profile_overhead.sh — compare the UCX-direct and CPU-proxy variants of the
# device pingpong benchmark.
#
# Modes:
#   sweep    [iters] [warmup]                   msg-size sweep on both binaries
#   nsys     [size]  [iters]   [warmup]         capture an Nsight Systems trace
#   ucxinfo  [size]  [iters]   [warmup]         dump UCX_PROTO_INFO for both
#   all                                         sweep + nsys + ucxinfo (defaults)
#
# Examples:
#   ./profile_overhead.sh sweep
#   ./profile_overhead.sh sweep 5000 500
#   ./profile_overhead.sh nsys 8192 2000
#   ./profile_overhead.sh ucxinfo 8 200
#   OUT_DIR=/tmp/run1 ./profile_overhead.sh all
#
# Tunables (env vars):
#   REPO_ROOT   path to nixl repo root            (default: derived from script)
#   BUILD_DIR   meson build dir                   (default: $REPO_ROOT/build)
#   BIN_DIR     directory holding both binaries   (default: $BUILD_DIR/examples/device/pingpong)
#   RECV_GPU    GPU id for receiver               (default: 0)
#   SEND_GPU    GPU id for sender                 (default: 0; safe to share one GPU)
#   RECV_HOST   host receiver listens on          (default: 127.0.0.1)
#   BASE_PORT   first TCP port to use             (default: 19500)
#   SIZES       space-separated msg sizes         (default: 8 64 512 4096 32768 262144 1048576)
#   USE_WARP    if 1, pass --warp                 (default: 0)
#   OUT_DIR     output directory                  (default: $REPO_ROOT/profile_results/<ts>)
#
# Notes:
# - Both binaries already encode their library paths via meson build_rpath, so
#   we don't need LD_LIBRARY_PATH.
# - Two-process loopback is required by the proxy variant.  We run sender and
#   receiver on the same host, on different GPUs, listening on different ports.

set -euo pipefail

# ---------- locate repo / binaries -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/examples/device/pingpong}"

UCX_BIN="${BIN_DIR}/nixl_device_pingpong"
PROXY_BIN="${BIN_DIR}/nixl_device_pingpong_proxy"

for bin in "${UCX_BIN}" "${PROXY_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "ERROR: missing binary ${bin}" >&2
        echo "Build with: ninja -C ${BUILD_DIR} examples/device/pingpong/nixl_device_pingpong examples/device/pingpong/nixl_device_pingpong_proxy" >&2
        exit 1
    fi
done

# ---------- defaults ---------------------------------------------------------
RECV_GPU="${RECV_GPU:-0}"
SEND_GPU="${SEND_GPU:-0}"
RECV_HOST="${RECV_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-19500}"
USE_WARP="${USE_WARP:-0}"
SIZES_STR="${SIZES:-8 64 512 4096 32768 262144 1048576}"
read -ra SIZES_ARR <<< "${SIZES_STR}"

DEFAULT_ITERS=2000
DEFAULT_WARMUP=200

# Quiet down the binaries: bench_host.cpp spins on prepMemView during setup
# until remote metadata is loaded; without throttling it can emit MB/s of
# ERROR logs.  FATAL keeps the genuinely fatal logs but suppresses the spam.
# Override by exporting NIXL_LOG_LEVEL before running the script.
export NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-FATAL}"

# Force the proxy worker's per-stage stats on by default.  The C++ predicate
# accepts any value other than the explicit disable shortcuts; we set "1" so
# the [proxy-stats] lines reliably end up in the *_send.err / *_recv.err
# files that analyze_nsys.sh consumes.  Set NIXL_PROXY_STATS=0 to disable.
export NIXL_PROXY_STATS="${NIXL_PROXY_STATS:-1}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/profile_results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT_DIR}"

WARP_FLAG=""
[[ "${USE_WARP}" == "1" ]] && WARP_FLAG="--warp"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Pick a fresh free port pair per run.  `ss -tln` lists current LISTEN sockets;
# we walk forward from BASE_PORT until we find two free consecutive ports.
# This avoids collisions with stale procs (use `pkill -f nixl_device_pingpong`
# to clean those up) and TIME_WAIT from prior runs.
_port_cursor="${BASE_PORT}"
next_port_pair() {
    local in_use
    in_use=$(ss -tlnH 2>/dev/null | awk '{ n=split($4,a,":"); print a[n] }' | sort -u)
    while :; do
        local p_recv="${_port_cursor}"
        local p_send=$((_port_cursor + 1))
        _port_cursor=$((_port_cursor + 2))
        if ! grep -qx "${p_recv}\|${p_send}" <<< "${in_use}"; then
            echo "${p_recv} ${p_send}"
            return 0
        fi
        if (( _port_cursor > BASE_PORT + 200 )); then
            log "ERROR: no free port pair found in [${BASE_PORT}, ${_port_cursor}]"
            return 1
        fi
    done
}

# Kill leftover bench processes (use --kill-stale flag or KILL_STALE=1).
maybe_kill_stale() {
    [[ "${KILL_STALE:-0}" != "1" ]] && return 0
    local victims
    victims=$(pgrep -f 'nixl_device_pingpong(_proxy)?( |$)' 2>/dev/null || true)
    if [[ -n "${victims}" ]]; then
        log "killing stale bench processes: $(echo ${victims} | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill -9 ${victims} 2>/dev/null || true
        sleep 1
    fi
}

# ---------- one (binary, size, iters) point ----------------------------------
# Spawns receiver in background, sender in foreground.  Echoes sender stdout.
# $1 binary  $2 size  $3 iters  $4 warmup  $5 tag  [$6 nsys-rep base]
run_one() {
    local bin="$1" size="$2" iters="$3" warmup="$4" tag="$5"
    local nsys_rep="${6:-}"

    local recv_out="${OUT_DIR}/${tag}_recv.out"
    local recv_err="${OUT_DIR}/${tag}_recv.err"
    local send_out="${OUT_DIR}/${tag}_send.out"
    local send_err="${OUT_DIR}/${tag}_send.err"
    read -r p_recv p_send < <(next_port_pair)

    log "  run tag=${tag} size=${size} iters=${iters} warmup=${warmup} ports=recv:${p_recv}/send:${p_send}"

    "${bin}" --role receiver --gpu "${RECV_GPU}" \
             --listen-port "${p_recv}" \
             --peer-ip "${RECV_HOST}" --peer-port "${p_send}" \
             --msg-size "${size}" --iters "${iters}" --warmup "${warmup}" \
             ${WARP_FLAG} \
             >"${recv_out}" 2>"${recv_err}" &
    local recv_pid=$!

    sleep 1

    local sender_cmd=( "${bin}" --role sender --gpu "${SEND_GPU}"
                       --listen-port "${p_send}"
                       --peer-ip "${RECV_HOST}" --peer-port "${p_recv}"
                       --msg-size "${size}" --iters "${iters}" --warmup "${warmup}" )
    [[ -n "${WARP_FLAG}" ]] && sender_cmd+=( "${WARP_FLAG}" )

    local rc=0
    if [[ -n "${nsys_rep}" ]]; then
        nsys profile -t cuda,nvtx,osrt -o "${nsys_rep}" --force-overwrite=true \
            "${sender_cmd[@]}" >"${send_out}" 2>"${send_err}" || rc=$?
    else
        "${sender_cmd[@]}" >"${send_out}" 2>"${send_err}" || rc=$?
    fi

    # Wait for receiver, but cap to RECV_WAIT_S so a hung peer can't make the
    # script hang silently.  If it's still alive, kill it.
    local waited=0
    while kill -0 "${recv_pid}" 2>/dev/null; do
        if (( waited >= ${RECV_WAIT_S:-30} )); then
            log "    receiver pid=${recv_pid} still alive after ${waited}s — killing"
            kill -9 "${recv_pid}" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    wait "${recv_pid}" 2>/dev/null || true

    if (( rc != 0 )); then
        log "    sender FAILED (rc=${rc}) — see ${send_out} ${send_err} ${recv_out} ${recv_err}"
        return "${rc}"
    fi
    cat "${send_out}"
}

# Parse "RTT=4.347 us" from sender stdout.
parse_rtt_us() {
    awk 'match($0, /RTT=([0-9.]+)[[:space:]]*us/, m) { print m[1]; exit }'
}

# ---------- mode: sweep ------------------------------------------------------
do_sweep() {
    local iters="${1:-${DEFAULT_ITERS}}"
    local warmup="${2:-${DEFAULT_WARMUP}}"
    local csv="${OUT_DIR}/sweep.csv"

    log "sweep iters=${iters} warmup=${warmup} sizes=(${SIZES_ARR[*]})"
    echo "variant,msg_size,iters,warmup,rtt_us" >"${csv}"

    for size in "${SIZES_ARR[@]}"; do
        for variant in ucx proxy; do
            local bin
            [[ "${variant}" == "ucx" ]] && bin="${UCX_BIN}" || bin="${PROXY_BIN}"
            local tag="sweep_${variant}_${size}"
            local out=""
            if out=$(run_one "${bin}" "${size}" "${iters}" "${warmup}" "${tag}"); then
                local rtt
                rtt=$(echo "${out}" | parse_rtt_us)
                [[ -z "${rtt}" ]] && rtt="NaN"
                echo "${variant},${size},${iters},${warmup},${rtt}" >>"${csv}"
                log "    ${variant} size=${size} -> ${rtt} us"
            else
                echo "${variant},${size},${iters},${warmup},FAIL" >>"${csv}"
            fi
            sleep 1
        done
    done

    log "wrote ${csv}"
    print_sweep_summary "${csv}"
}

print_sweep_summary() {
    local csv="$1"
    local txt="${OUT_DIR}/summary.txt"
    {
        echo "Sweep summary  (csv: ${csv})"
        echo "---------------------------------------------------------------------------"
        awk -F, '
            NR==1 { next }
            { rtt[$1","$2]=$5; sizes[$2]=1 }
            END {
                printf "  %10s  %12s  %12s  %12s  %10s\n",
                       "msg_size", "ucx_us", "proxy_us", "delta_us", "ratio"
                n=asorti(sizes, ks, "@ind_num_asc")
                for (i=1;i<=n;i++) {
                    s=ks[i]
                    u=rtt["ucx,"s]+0; p=rtt["proxy,"s]+0
                    if (u>0 && p>0) {
                        printf "  %10d  %12.2f  %12.2f  %12.2f  %9.2fx\n",
                               s, u, p, p-u, p/u
                    } else {
                        printf "  %10d  %12s  %12s  %12s  %10s\n",
                               s,
                               (u>0)?sprintf("%.2f",u):"FAIL",
                               (p>0)?sprintf("%.2f",p):"FAIL",
                               "-", "-"
                    }
                }
            }' "${csv}"
    } | tee "${txt}"
    log "wrote ${txt}"
}

# ---------- mode: nsys -------------------------------------------------------
do_nsys() {
    local size="${1:-8192}"
    local iters="${2:-2000}"
    local warmup="${3:-${DEFAULT_WARMUP}}"

    if ! command -v nsys >/dev/null; then
        log "nsys not on PATH — skipping nsys mode"
        return 1
    fi

    log "nsys size=${size} iters=${iters} warmup=${warmup}"
    for variant in ucx proxy; do
        local bin
        [[ "${variant}" == "ucx" ]] && bin="${UCX_BIN}" || bin="${PROXY_BIN}"
        local tag="nsys_${variant}_${size}"
        local rep="${OUT_DIR}/${tag}"
        run_one "${bin}" "${size}" "${iters}" "${warmup}" "${tag}" "${rep}" >/dev/null || true
        log "  wrote ${rep}.nsys-rep"
    done
    log "open the .nsys-rep files in Nsight Systems to compare timelines"
}

# ---------- mode: ucxinfo ----------------------------------------------------
do_ucxinfo() {
    local size="${1:-8}"
    local iters="${2:-200}"
    local warmup="${3:-50}"

    log "ucxinfo size=${size} iters=${iters} warmup=${warmup}"
    for variant in ucx proxy; do
        local bin
        [[ "${variant}" == "ucx" ]] && bin="${UCX_BIN}" || bin="${PROXY_BIN}"
        local tag="ucxinfo_${variant}"
        log "  ${variant}: capturing UCX_PROTO_INFO"
        UCX_LOG_LEVEL=info UCX_PROTO_INFO=y \
            run_one "${bin}" "${size}" "${iters}" "${warmup}" "${tag}" >/dev/null || true
        log "    sender log: ${OUT_DIR}/${tag}_send.log"
        log "    recv   log: ${OUT_DIR}/${tag}_recv.log"
    done
}

# ---------- entrypoint -------------------------------------------------------
mode="${1:-sweep}"; shift || true

# Preflight: detect leftover bench processes (common cause of port-in-use
# failures across runs).  Auto-kill if KILL_STALE=1 was set.
maybe_kill_stale
stale=$(pgrep -f 'nixl_device_pingpong(_proxy)?( |$)' 2>/dev/null || true)
if [[ -n "${stale}" ]]; then
    log "WARNING: existing bench processes are running (pids: $(echo ${stale} | tr '\n' ' '))"
    log "         re-run with KILL_STALE=1 to auto-kill, or:  pkill -9 -f nixl_device_pingpong"
fi

case "${mode}" in
    sweep)    do_sweep    "$@" ;;
    nsys)     do_nsys     "$@" ;;
    ucxinfo)  do_ucxinfo  "$@" ;;
    all)
        do_sweep
        do_nsys 8192 2000
        do_ucxinfo 8 200
        ;;
    -h|--help|help)
        sed -n '2,30p' "$0"
        exit 0
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        sed -n '2,30p' "$0" >&2
        exit 2
        ;;
esac

log "results in ${OUT_DIR}"
