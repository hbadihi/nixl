# Evals: NIXL EP dispatch over the LIBFABRIC device proxy, one p5 node

**This file is the source of truth for how we evaluate. Do not edit it casually.**
- It holds definitions only: commands, metrics, pass rules. Results and notes go elsewhere.
- Never tune an eval's flags to improve its number. A changed flag is a new eval ID.
- To change an eval: add a line to the Changelog at the bottom and re-baseline HEAD under the new definition.
  Numbers from before and after a definition change must never be compared.

**Goal:** NIXL EP dispatch per GPU approaches the nixlbench ceiling (~30 GB/s). The E2 target is set after its first baseline.
Code: branch `efa-proxy/1t-perf-clean` (`8bac28bb`) plus the `--dispatch-only` flag (`5a3f0944`, branch `worktree-ep-dispatch-only`).

## Setup (every run)

```bash
export NIXL=<checkout under test, built with install/ and install-ep/>
export LD_LIBRARY_PATH=$NIXL/install/lib:$NIXL/install/lib/x86_64-linux-gnu:/workspace/external/libfabric-rc65/install/lib:/workspace/external/rdma-core/install/lib:/usr/local/cuda/lib64
export NIXL_PLUGIN_DIR=$NIXL/install/lib/x86_64-linux-gnu/plugins
export FI_EFA_ENABLE_SHM_TRANSFER=0   # same host: use the EFA NICs, not shm
```
For EP replace `install` with `install-ep`.

Preconditions (record whether each held; a run where one did not is void):
- libfabric built with gdrcopy.
- `taskset` as given below; without it EP runs at about half speed.
- No spinning kworker in the taskset CPU range, no other job on the GPUs or NICs.
- No leftover workers: `pkill -9 -f "^/usr/bin/python3 -c from multiprocessing[.]spawn"`.

## Evals

### E1: nixlbench 1 GPU -> 1 GPU, 8 KiB puts (reference ceiling, ~30 GB/s)

Run for reference only; it has no pass rule. Metric: bandwidth.

```bash
NB="/workspace/external/nixl/benchmark/nixlbench/install/bin/nixlbench --runtime_type ASIO --asio_address 127.0.0.1 --asio_port 12345 \
  --backend LIBFABRIC --initiator_seg_type VRAM --target_seg_type VRAM --use_device_api --device_mode=proxy \
  --benchmark_group efa_gpu --num_threads=32 --start_batch_size=1024 --max_batch_size=1024 \
  --start_block_size=8192 --max_block_size=8192 --num_iter 10000 --device_channel_num=4 --device_proxy_threads=4"
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23  $NB   # server
CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 $NB   # client
```

### E2: NIXL EP dispatch only, 4 ranks, all-to-all (the main eval)

One process group, 4 ranks on GPUs 0-3 (NUMA 0), 128 experts (32 per rank), 128 tokens, hidden 7168, topk 8. Dispatch only; combine is skipped.

```bash
cd $NIXL/examples/device/ep
export PYTHONPATH=/workspace/external/ep-pylib:$NIXL/install-ep/lib/python3/dist-packages:$NIXL/src/bindings/python/nixl-meta
export NIXL_EP_BACKEND=LIBFABRIC NIXL_EP_DEVICE_MODE=proxy
taskset -c 0-47,96-143 python3 tests/elastic/elastic.py --plan /workspace/external/nixl/.runlogs/hod/ep_4ranks.json \
  --num-processes 4 --num-experts-per-rank 32 --disable-ll-nvlink --dispatch-only
```
Output per rank: `[rank N] Dispatch bandwidth: X GB/s, avg_t=... us, min_t=... us, max_t=... us`.

## What to read, in this order

Every E2 run reports all of these. Each answers a different question.

| # | Metric | Question it answers | Rule |
|---|---|---|---|
| 1 | **Dispatch avg bandwidth (GB/s)**: mean over the 4 ranks of the per-rank `avg_t` bandwidth | Did it get faster? | **Main metric.** Higher is better. |
| 2 | **min_t and max_t (us)** per iteration, per rank (max across ranks reported) | Is it stable? | Secondary. Variance is bad: a large max_t/avg_t is a regression even if the average holds. |
| 3 | **nixlPut time in the kernel (us): p50, p90, p99** | How long does a GPU warp spend issuing one put? | Timed in the kernel before and after the call. p99 vs p50 shows the tail. |
| 4 | **nixlAtomicAdd time in the kernel (us): p50, p90, p99** | Same for the signal. | Same method as #3. |
| 5 | **Ring-full waits: count and total wait time** | Did the GPU ever wait for a ring slot before it could submit? | **Red flag: any non-zero count.** The run is flagged until it is explained or fixed. |

Notes:
- Compare p50, p90 and p99 separately; a change that moves only p99 is a tail effect, not a throughput effect.
- Report the main metric together with #2 and #5; a faster average with a worse max_t or any ring-full wait is not a win.
- Instrumentation status: #1 and #2 are printed today. #3, #4 and #5 are **not instrumented yet**; report them as `n/a` until they are added.
  Adding them is a code change, not an eval change, and must not alter timing when disabled.

## Correctness gates

A performance number counts only if these pass.
- **C1:** a default EP run (without `--dispatch-only`, same command otherwise) passes the test's built-in checks. It checks the dispatched and combined data.
- **C2:** the E2 run itself finishes with no hang, no timeout, a clean exit, and no leftover workers.

## Recording a run

One row per run, in a results file (not here): date, commit, eval ID, preconditions held (y/n), metrics #1-#5, C1/C2 result.
Repeat each eval 5 times; report the median of #1 and the worst of #2 and #5.

## Changelog

- Defined E1 and E2 (4 ranks, dispatch only) and metrics #1-#5. E2 replaces the earlier 8-rank dispatch+combine run.
