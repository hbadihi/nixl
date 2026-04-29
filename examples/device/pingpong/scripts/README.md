# pingpong profiling scripts

## profile_overhead.sh

Measures the latency overhead of the CPU-proxy backend vs. UCX-direct using
the two pingpong binaries built from this directory:

- `nixl_device_pingpong`         — UCX direct (GDA-KI / DOCA path)
- `nixl_device_pingpong_proxy`   — CPU proxy enabled

The proxy build does not support single-process mode (the proxy device
context is published to a process-wide `__device__` pointer), so the script
always runs two-process loopback on a single host. By default both processes
use GPU 0; if you have multiple GPUs, set `SEND_GPU=1` to put the sender on a
different device.

### Prerequisites

```bash
ninja -C build examples/device/pingpong/nixl_device_pingpong \
                  examples/device/pingpong/nixl_device_pingpong_proxy
```

The script auto-discovers binaries under `build/examples/device/pingpong/`
relative to the script location. Override with `BUILD_DIR=` or `BIN_DIR=`.

### Usage

```bash
# Default sweep across small..large messages on both binaries
./scripts/profile_overhead.sh sweep

# Tune iters / warmup
./scripts/profile_overhead.sh sweep 5000 500

# Capture an Nsight Systems trace at one point
./scripts/profile_overhead.sh nsys 8192 2000

# Dump UCX protocol selection
./scripts/profile_overhead.sh ucxinfo

# All three with defaults
./scripts/profile_overhead.sh all
```

### Output

Each invocation creates a fresh timestamped directory under
`profile_results/` (override with `OUT_DIR=`):

| File                                          | Contents                                       |
|-----------------------------------------------|------------------------------------------------|
| `sweep.csv`                                   | `variant,msg_size,iters,warmup,rtt_us`         |
| `summary.txt`                                 | side-by-side RTT, delta, ratio per size        |
| `sweep_<variant>_<size>_{send,recv}.out`      | sender / receiver stdout (RTT line)            |
| `sweep_<variant>_<size>_{send,recv}.err`      | sender / receiver stderr (NIXL/CUDA logs)      |
| `nsys_<variant>_<size>.nsys-rep`              | Nsight Systems traces                          |
| `ucxinfo_<variant>_{send,recv}.{out,err}`     | UCX_PROTO_INFO output per variant              |

Stdout and stderr are intentionally split so the (very small) RTT print isn't
buried in NIXL logs, and so a noisy backend can't blow up the file we parse.

The script defaults `NIXL_LOG_LEVEL=FATAL` to keep the `*.err` files small.
Override to debug:

```bash
NIXL_LOG_LEVEL=INFO ./scripts/profile_overhead.sh sweep
```

### Tunables (environment variables)

| Var               | Default                                                     |
|-------------------|-------------------------------------------------------------|
| `RECV_GPU`        | `0`                                                         |
| `SEND_GPU`        | `0` (defaults to same GPU; set `SEND_GPU=1` if you have ≥2) |
| `RECV_HOST`       | `127.0.0.1`                                                 |
| `BASE_PORT`       | `19500` — first port to *try*; the script auto-walks past   |
|                   | already-bound ports (via `ss -tln`) to find a free pair     |
| `SIZES`           | `8 64 512 4096 32768 262144 1048576`                        |
| `USE_WARP`        | `0` — set to `1` to pass `--warp`                           |
| `KILL_STALE`      | `0` — set to `1` to auto-kill leftover bench procs at start |
| `RECV_WAIT_S`     | `30` — max seconds to wait for the receiver before killing  |
| `NIXL_LOG_LEVEL`  | `FATAL` — silences `prepMemView` setup-loop noise           |
| `NIXL_PROXY_STATS`| `1` — exported by the script so `[proxy-stats]` reliably    |
|                   | lands in the `*_send.err` / `*_recv.err` files.  Set to `0` |
|                   | (or `false`/`off`/`no`) to disable.                         |
| `BUILD_DIR`       | `<repo>/build`                                              |
| `BIN_DIR`         | `$BUILD_DIR/examples/device/pingpong`                       |
| `OUT_DIR`         | `<repo>/profile_results/<timestamp>`                        |

### Recovering from stuck runs

If a previous invocation left bench processes alive (you'll see a
`port 19500 in use?` error or the script's own `WARNING: existing bench
processes are running` line), clean them up:

```bash
sudo pkill -9 -f nixl_device_pingpong
# or let the script do it next time:
KILL_STALE=1 ./scripts/profile_overhead.sh sweep
```

If you ran the script with `sudo` previously, the output dirs are root-owned
and need `sudo rm -rf profile_results/`.

### Reading the summary

`summary.txt` looks like:

```
  msg_size       ucx_us      proxy_us      delta_us       ratio
         8         4.35         13.05          8.70       3.00x
        64         4.40         13.10          8.70       2.98x
      4096         4.80         13.60          8.80       2.83x
   1048576        45.10         55.20         10.10       1.22x
```

If `delta_us` is roughly constant across sizes, the cost is dominated by the
fixed proxy hop (GPU enqueue → worker poll → UCX submit → completion publish).
If it grows with size, look at `prepareSubmission` and the UCX submit path in
`src/core/device_proxy/proxy_worker.cpp`.

### Per-stage proxy-worker stats

`ProxyWorker` is instrumented with per-stage timing. At thread shutdown each
worker prints lines like:

```
[proxy-stats][w0] runOnce_iters=1234567  progress_calls=1234567
[proxy-stats][w0] prep+submit  n=2200     avg=    8.412 us  min=    3.117 us  max=  142.001 us
[proxy-stats][w0] inflight     n=2200     avg= 4350.110 us  min= 4280.992 us  max= 4901.337 us
[proxy-stats][w0] publish      n=2200     avg=    1.108 us  min=    0.523 us  max=   84.711 us
[proxy-stats][w0] polls/request=560.0  (1.0 == every poll dispatched; high == worker spinning idle)
```

How to read it:

| Stage          | What it measures                                          | Typical low value |
|----------------|-----------------------------------------------------------|-------------------|
| `prep+submit`  | from CPU dequeuing the GPU's request to `backend->submit` returning (covers `prepareSubmission` and the UCX `prepXfer + postXfer`) | a few µs          |
| `inflight`     | from submit return to seeing the completion (this is the cost of UCX itself) | µs on RDMA, ms on TCP |
| `publish`      | from observing the completion to writing it back to the GPU completion slot | sub-µs            |
| `polls/request`| how many `runOnce` iterations the worker spun for each request it handled. High numbers (10+) mean the worker is spinning on a cold ring most of the time. |  |

If `inflight` dominates the total proxy delta from the sweep summary (`delta_us`),
the cost is in UCX/network — improving the proxy itself won't help much. If
`prep+submit` dominates, the proxy submission path is the bottleneck.

These stats are written to **stderr regardless of `NIXL_LOG_LEVEL`**, so they
appear in `*_send.err` / `*_recv.err`. The script exports `NIXL_PROXY_STATS=1`
on your behalf; disable with `NIXL_PROXY_STATS=0`. The predicate also accepts
`false`/`off`/`no` (case-insensitive) as disable values.

The `[proxy-stats]` block is only emitted at worker-thread shutdown (when the
agent is destroyed), so a `.err` file inspected while the run is still
in progress will not yet contain it. Wait for the run to finish (or for the
matching `_send.out` to contain its `RTT=` line) before grepping.

### NVTX ranges in the proxy worker

`ProxyWorker` is instrumented with NVTX 3 ranges (header-only, no link
dependency — they're near-free when no profiler is attached). When you capture
with `profile_overhead.sh nsys`, the resulting `.nsys-rep` will contain:

| Range          | Bracketed code                                                         |
|----------------|------------------------------------------------------------------------|
| `prx:submit`   | per-op `submitToBackend` (prepareSubmission + `backend->submit`)       |
| `prx:progress` | `backend->progress()`, only when at least one channel has in-flight work (skipped during pure-idle spinning to keep the report sane) |
| `prx:complete` | mark — instant where the worker first observed a completion           |
| `prx:publish`  | atomic store of `completed_idx` back to the GPU completion slot        |

These ranges show up in both Nsight Systems' GUI timeline and in the
`nvtx_pushpop_sum` / `nvtx_pushpop_trace` reports of `nsys stats`.

## analyze_nsys.sh

Turns a capture directory (or a single `.nsys-rep`) into a one-shot text
report: sweep summary, per-capture RTT, `[proxy-stats]` lines, and the most
useful `nsys stats` reports.

```bash
# Most recent run under profile_results/
./scripts/analyze_nsys.sh

# A specific run dir (everything under it is analyzed)
./scripts/analyze_nsys.sh profile_results/20260423-112226

# A single capture
./scripts/analyze_nsys.sh profile_results/20260423-112226/nsys_proxy_8192.nsys-rep
```

Output is written to `<run_dir>/analysis.txt` and also printed to stderr (the
script echoes the file path on stdout so it composes with shell pipelines).

### Tunables

| Var            | Default                                                                                |
|----------------|----------------------------------------------------------------------------------------|
| `TOP_N`        | `10` rows kept per nsys report                                                         |
| `REPORTS`      | `nvtx_pushpop_sum cuda_api_sum cuda_gpu_kern_sum osrt_sum cuda_gpu_mem_time_sum`       |
| `OUT_FILE`     | `<run_dir>/analysis.txt`                                                               |
| `RESULTS_ROOT` | `<repo>/profile_results`                                                               |

### What to look for

1. `nvtx_pushpop_sum` should show `prx:progress` taking the bulk of the worker
   thread's time when in-flight work exists — that's UCX waiting for the
   completion. `prx:submit` and `prx:publish` should be small.
2. Diff the `cuda_api_sum` between the UCX-direct and proxy captures; new
   high-volume entries on the proxy side (`cuStreamAddCallback`, `cuEventQuery`)
   are the cost of host↔device coordination.
3. Cross-check the `inflight` value from `[proxy-stats]` against the
   sweep `delta_us`. If `inflight` ≈ `delta_us`, the bottleneck is UCX
   itself, not the proxy bookkeeping.

### Going further

1. Add NVTX ranges around the GPU-side `do_put_async` path in
   `bench_kernel.cu` (they need to be NVTX-CUDA, not regular NVTX — see
   `<nvtx3/nvToolsExtCuda.h>`).
2. Bracket `do_put_async` and the wait loop in `bench_kernel.cu` with
   `clock64()` to attribute time inside the GPU kernel.
3. Re-export captures to SQLite (`nsys export -t sqlite foo.nsys-rep`) and
   query with `sqlite3` for fully programmatic analysis.
