#!/usr/bin/env bash
#
# Sweep proxy worker configurations for the elastic LL test and summarize pass/fail.
#
# Two INDEPENDENT axes (wired in Buffer::_nixl_agent_init, examples/device/ep/csrc/nixl_ep.cpp):
#   NIXL_EP_PROXY_UCX_WORKERS   UCX host workers  -> one EP/QP/rkey per worker per peer.
#                               A submission routes to worker (channel_id % num_workers), so
#                               this is QP-level parallelism per peer. (default = lane ceiling)
#   NIXL_EP_PROXY_WORKER_COUNT  proxy drain threads that consume the GPU rings. (default = 1)
#
# channels_per_rank (the device ring stride / per-rank isolation) is fixed at the lane ceiling
# and is independent of both knobs, so any (ucx_workers, proxy_workers) combo is correctness-safe;
# they only change which QP carries a channel and how many CPU threads drain.
#
# Usage (run from the EP example dir, with PYTHONPATH at your PROXY build tree):
#   export PYTHONPATH=<repo>/build-proxy/examples/device/ep
#   cd <repo>/examples/device/ep
#   bash tests/elastic/proxy_worker_matrix.sh
#
# Override anything via env:
#   UCX_WORKERS="1 2 4 8" PROXY_WORKERS="1 2" \
#   PLAN=tests/elastic/single_expansion.json NUM_PROCESSES=8 \
#   TIMEOUT_MS=30000 CELL_WALL_TIMEOUT=180 OUT_DIR=/tmp/ep_proxy_matrix \
#   bash tests/elastic/proxy_worker_matrix.sh
#
# Notes:
#  - TIMEOUT_MS is the GPU/kernel dispatch-receive timeout; keep it modest (e.g. 30s) so a
#    stalled cell fails fast instead of hanging. CELL_WALL_TIMEOUT is an OS-level backstop
#    (SIGINT) that kills a truly hung cell so the sweep keeps moving.
#  - proxy_workers > 1 makes multiple drain threads hit the shared UCX engine concurrently;
#    that MT path is not yet validated, so treat W>1 results as exploratory.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

UCX_WORKERS="${UCX_WORKERS:-1 2 4 8}"
PROXY_WORKERS="${PROXY_WORKERS:-1}"
PLAN="${PLAN:-tests/elastic/expansion_contraction.json}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TIMEOUT_MS="${TIMEOUT_MS:-30000}"
CELL_WALL_TIMEOUT="${CELL_WALL_TIMEOUT:-180}"
OUT_DIR="${OUT_DIR:-/tmp/ep_proxy_matrix}"
LOG_LEVEL="${NIXL_LOG_LEVEL:-INFO}"

if [[ "$PLAN" = /* ]]; then
    PLAN_PATH="$PLAN"
else
    PLAN_PATH="$EP_DIR/$PLAN"
fi

# Pre-flight: confirm the EP extension is importable (PYTHONPATH points at a built tree).
if ! python3 -c "import nixl_ep; print('nixl_ep backend:', nixl_ep.get_gpu_device_api_backend())" 2>/tmp/_ep_import.txt; then
    echo "ERROR: cannot import nixl_ep. Set PYTHONPATH to <build>/examples/device/ep and run from the EP example dir." >&2
    cat /tmp/_ep_import.txt >&2
    exit 1
fi
backend=$(python3 -c "import nixl_ep; print(nixl_ep.get_gpu_device_api_backend())" 2>/dev/null)
if [ "$backend" != "proxy" ]; then
    echo "WARNING: nixl_ep backend is '$backend', not 'proxy'. The worker knobs only affect the proxy build." >&2
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "ucx_workers\tproxy_workers\tresult\texit\tdispatch_timeouts\tfailure_detections\tevidence_accepted\tevidence_files\tlog\n" > "$SUMMARY"

echo "Matrix: UCX_WORKERS=[$UCX_WORKERS] x PROXY_WORKERS=[$PROXY_WORKERS]"
echo "  plan=$PLAN_PATH procs=$NUM_PROCESSES timeout_ms=$TIMEOUT_MS wall=${CELL_WALL_TIMEOUT}s backend=$backend"
echo "  output: $OUT_DIR"
echo

for uw in $UCX_WORKERS; do
  for pw in $PROXY_WORKERS; do
    cell="uw${uw}_pw${pw}"
    log="$OUT_DIR/$cell.log"
    ev="$OUT_DIR/$cell.evidence"
    rm -rf "$ev"; mkdir -p "$ev"
    echo "=== cell $cell : NIXL_EP_PROXY_UCX_WORKERS=$uw NIXL_EP_PROXY_WORKER_COUNT=$pw ==="

    NIXL_EP_PROXY_UCX_WORKERS="$uw" \
    NIXL_EP_PROXY_WORKER_COUNT="$pw" \
    NIXL_LOG_LEVEL="$LOG_LEVEL" \
      timeout -s INT "$CELL_WALL_TIMEOUT" \
        python3 "$SCRIPT_DIR/elastic.py" \
          --plan "$PLAN_PATH" \
          --num-processes "$NUM_PROCESSES" \
          --timeout-ms "$TIMEOUT_MS" \
          --evidence-output "$ev" \
      > "$log" 2>&1
    rc=$?

    dt=$(grep -c "NIXL-EP timeout for dispatch receive" "$log")
    fd=$(grep -c "detected unexpected rank failures" "$log")
    evfiles=$(find "$ev" -maxdepth 1 -type f | wc -l | tr -d ' ')
    evacc=$(grep -rl '"classification": "accepted"' "$ev" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$rc" -eq 0 ]; then
        result="PASS"
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 137 ]; then
        result="HANG_WALL_TIMEOUT"
    else
        result="FAIL"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$uw" "$pw" "$result" "$rc" "$dt" "$fd" "$evacc" "$evfiles" "$log" >> "$SUMMARY"
    echo "    -> $result (exit=$rc dispatch_timeouts=$dt failure_detections=$fd evidence_accepted=$evacc/$evfiles)"

    # Let TCPStore/rank-server ports free and GPUs settle before the next cell.
    sleep 5
  done
done

echo
echo "================ SUMMARY ================"
if command -v column >/dev/null 2>&1; then
    column -t -s "$(printf '\t')" "$SUMMARY"
else
    cat "$SUMMARY"
fi
echo
echo "Per-cell logs + evidence under $OUT_DIR ; machine-readable summary: $SUMMARY"
echo "Legend: PASS=plan completed (exit 0); HANG_WALL_TIMEOUT=killed by ${CELL_WALL_TIMEOUT}s backstop;"
echo "        FAIL=workers errored. dispatch_timeouts/failure_detections are grep counts from the log."
