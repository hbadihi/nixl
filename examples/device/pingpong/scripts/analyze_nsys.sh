#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# analyze_nsys.sh — turn a profile_overhead.sh capture directory (or a single
# .nsys-rep file) into a single text report that answers the question
# "where does the proxy hop go?".
#
# It runs a curated subset of `nsys stats` reports plus pulls the
# `[proxy-stats]` lines that proxy_worker.cpp logs at shutdown.  When given a
# directory it pairs the UCX-direct and CPU-proxy captures side-by-side.
#
# Usage:
#   ./analyze_nsys.sh                                # most recent run
#   ./analyze_nsys.sh path/to/profile_results/<ts>   # specific run dir
#   ./analyze_nsys.sh path/to/foo.nsys-rep           # single capture
#
# Tunables (env vars):
#   REPO_ROOT      path to nixl repo root        (default: derived from script)
#   RESULTS_ROOT   where profile_overhead writes (default: $REPO_ROOT/profile_results)
#   TOP_N          rows kept per report          (default: 10)
#   REPORTS        space-separated nsys stats reports
#                  (default: nvtx_pushpop_sum cuda_api_sum cuda_gpu_kern_sum
#                            osrt_sum cuda_gpu_mem_time_sum)
#   OUT_FILE       where to write the analysis   (default: <run_dir>/analysis.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/profile_results}"
TOP_N="${TOP_N:-10}"
REPORTS="${REPORTS:-nvtx_pushpop_sum cuda_api_sum cuda_gpu_kern_sum osrt_sum cuda_gpu_mem_time_sum}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

if ! command -v nsys >/dev/null; then
    echo "ERROR: nsys not on PATH" >&2
    exit 1
fi

# ---------- argument resolution ---------------------------------------------
arg="${1:-}"
if [[ -z "${arg}" ]]; then
    # Pick the most recent timestamped run directory.
    if [[ ! -d "${RESULTS_ROOT}" ]]; then
        echo "ERROR: ${RESULTS_ROOT} does not exist; pass a path explicitly" >&2
        exit 1
    fi
    arg=$(ls -1dt "${RESULTS_ROOT}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
    if [[ -z "${arg}" ]]; then
        echo "ERROR: no run directories under ${RESULTS_ROOT}" >&2
        exit 1
    fi
    log "using most recent run: ${arg}"
fi

run_dir=""
captures=()
if [[ -d "${arg}" ]]; then
    run_dir="${arg}"
    while IFS= read -r f; do captures+=( "${f}" ); done \
        < <(ls -1 "${run_dir}"/*.nsys-rep 2>/dev/null | sort)
elif [[ -f "${arg}" && "${arg}" == *.nsys-rep ]]; then
    run_dir="$(dirname "${arg}")"
    captures=( "${arg}" )
else
    echo "ERROR: '${arg}' is neither a directory nor a .nsys-rep file" >&2
    exit 1
fi

if (( ${#captures[@]} == 0 )); then
    echo "ERROR: no .nsys-rep files found in ${run_dir}" >&2
    echo "       generate one with: ./profile_overhead.sh nsys" >&2
    exit 1
fi

OUT_FILE="${OUT_FILE:-${run_dir}/analysis.txt}"

# ---------- helpers ----------------------------------------------------------
section() {
    printf '\n========================================================================\n'
    printf '  %s\n' "$*"
    printf '========================================================================\n'
}

run_stats_report() {
    # $1 capture path  $2 report name
    local cap="$1" rpt="$2"
    # `nsys stats` writes a temporary sqlite next to the .nsys-rep on first
    # invocation; subsequent calls reuse it.  We strip the leading banner
    # lines and head -N the body so reports stay scannable.
    nsys stats --report "${rpt}" --format csv --quiet "${cap}" 2>/dev/null \
        | awk -v top="${TOP_N}" '
              /^\s*$/        { next }
              /^Generating/  { next }
              /^Processing/  { next }
              /^Exporting/   { next }
              /^SQLite/      { next }
              /^Using/       { next }
              /^WARNING/     { next }
              /^\*+$/        { next }
              /^Report/      { next }
              { lines[++n] = $0 }
              END {
                  if (n == 0) { print "  (no data)"; exit }
                  # Print header (line 1) + up to top data rows.
                  print lines[1]
                  cap = (n - 1 > top) ? top + 1 : n
                  for (i = 2; i <= cap; i++) print lines[i]
                  if (n - 1 > top) printf "  ... %d more rows trimmed\n", n - 1 - top
              }' \
        | column -t -s,
}

dump_proxy_stats() {
    # $1 capture path -- find the matching *.err file written alongside.
    local cap="$1"
    local base
    base="$(basename "${cap}" .nsys-rep)"
    local matched=0
    for role in send recv; do
        local errf="${run_dir}/${base}_${role}.err"
        if [[ -f "${errf}" ]]; then
            local hits
            hits=$(grep -E '^\[proxy-stats\]' "${errf}" 2>/dev/null || true)
            if [[ -n "${hits}" ]]; then
                printf '  %s (%s):\n' "$(basename "${errf}")" "${role}"
                echo "${hits}" | sed 's/^/    /'
                matched=1
            fi
        fi
    done
    if (( matched == 0 )); then
        echo "  (no [proxy-stats] lines found — was this a UCX-direct capture, or did the worker thread skip clean shutdown?)"
    fi
    # Explicit success: avoid set -e killing the script when the last statement
    # above is a (( ... )) test that evaluated to false.
    return 0
}

dump_rtt() {
    # $1 capture path -- pull the RTT line from the matching _send.out file.
    local cap="$1"
    local base
    base="$(basename "${cap}" .nsys-rep)"
    local outf="${run_dir}/${base}_send.out"
    if [[ -f "${outf}" ]]; then
        grep -E 'RTT=' "${outf}" 2>/dev/null | sed 's/^/  /' || echo "  (no RTT line)"
    else
        echo "  (no ${base}_send.out)"
    fi
}

# ---------- main report ------------------------------------------------------
{
    printf 'analyze_nsys.sh report\n'
    printf 'generated:    %s\n' "$(date)"
    printf 'run dir:      %s\n' "${run_dir}"
    printf 'captures:     %d\n' "${#captures[@]}"
    printf 'top rows/rpt: %s\n' "${TOP_N}"
    printf 'reports:      %s\n' "${REPORTS}"

    # Pull the sweep summary if present — gives high-level RTT comparison.
    if [[ -f "${run_dir}/summary.txt" ]]; then
        section "sweep summary (from summary.txt)"
        sed 's/^/  /' "${run_dir}/summary.txt"
    fi

    for cap in "${captures[@]}"; do
        section "$(basename "${cap}")"

        printf '\n-- measured RTT (sender stdout) --\n'
        dump_rtt "${cap}"

        printf '\n-- proxy-stats summary (from *.err) --\n'
        dump_proxy_stats "${cap}"

        for rpt in ${REPORTS}; do
            printf '\n-- nsys stats: %s --\n' "${rpt}"
            run_stats_report "${cap}" "${rpt}" | sed 's/^/  /'
        done
    done

    # Side-by-side hint when both UCX and proxy captures exist.
    ucx_caps=()
    prx_caps=()
    for cap in "${captures[@]}"; do
        case "$(basename "${cap}")" in
            *ucx*)   ucx_caps+=( "${cap}" ) ;;
            *proxy*) prx_caps+=( "${cap}" ) ;;
        esac
    done
    if (( ${#ucx_caps[@]} > 0 && ${#prx_caps[@]} > 0 )); then
        section "next steps"
        cat <<EOF
  - Compare 'cuda_api_sum' between UCX and proxy captures: anything that
    appears (or grows) on the proxy side is overhead introduced by the hop.
  - In 'nvtx_pushpop_sum', the prx:submit / prx:progress / prx:publish
    ranges should account for the proxy worker's CPU time.  prx:progress
    dominating means UCX is busy-polling waiting for the network.
  - Cross-check 'inflight' time from [proxy-stats] against the proxy RTT
    delta — if inflight dominates, the bottleneck is UCX itself, not the
    proxy bookkeeping, so optimizing the ring/worker won't help much.
  - Open the captures in the Nsight Systems GUI for a visual timeline:
      nsys-ui ${ucx_caps[0]} &
      nsys-ui ${prx_caps[0]} &
EOF
    fi
} | tee "${OUT_FILE}" >/dev/null

log "analysis written to ${OUT_FILE}"
echo "${OUT_FILE}"
