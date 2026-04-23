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
| `NIXL_PROXY_STATS`| `1` — set to `0` to disable per-stage proxy worker stats    |
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
appear in `*_send.err` / `*_recv.err`. Disable with `NIXL_PROXY_STATS=0`.

### Going further

1. Add NVTX ranges around each stage of `ProxyWorker::runOnce` and re-run
   `nsys` mode for visual confirmation in the timeline.
2. Bracket `do_put_async` and the wait loop in `bench_kernel.cu` with
   `clock64()` to attribute time inside the GPU kernel.
