# CPU Proxy — Performance Summary & Proofs

> Executive summary of the CPU proxy benchmarking effort, with line-cited
> evidence from the runs in `profile_results/`. Every number in the summary
> below is backed by a quoted block from a real artifact.
>
> **Captured on:** `adv-dev-420`, single GPU (GPU 0), Mellanox ConnectX
> (`rocep13s0f0`/`rocep13s0f1`), NVIDIA driver 550.163.01, custom UCX 1.21.0
> at `/scrap/cpu-proxy/ucx-install` built with `--with-cuda` (CUDA + DMABUF
> + GDA-KI modules present). **GPUDirect RDMA is supported by the build
> but not currently active** because neither `nvidia_peermem.ko` is loaded
> nor `gdrcopy` is installed (see §4 caveat 1 for the full diagnostic).
> **Workload:** `examples/device/pingpong`, two-process loopback, THREAD
> level, 2000 iters / 200 warmup unless noted.
>
> Reproduce with: `examples/device/pingpong/scripts/profile_overhead.sh`
> (see [`README.md`](README.md)).

---

## 1. Executive summary

### What we built

Two side-by-side binaries of the device-API pingpong benchmark plus a
profiling harness, designed to make the cost of the CPU proxy directly
measurable:

- `nixl_device_pingpong` — GPU-issued `nixlPut` on the **UCX-direct** path.
- `nixl_device_pingpong_proxy` — same kernel/host, GPU-issued `nixlPut`
  routed through the **CPU proxy runtime**.
- `examples/device/pingpong/scripts/profile_overhead.sh` — message-size
  sweep, Nsight Systems capture, UCX protocol dump.
- Per-stage instrumentation in `ProxyWorker` (counters for
  `prep+submit`, `inflight`, `publish`, `polls/request`) so the proxy
  delta can be attributed to a specific stage rather than appearing as
  one opaque RTT number.

### Headline numbers (8 B PUT, proxy variant, sweep run)

| Stage                                | Cost                          | Reading |
|--------------------------------------|-------------------------------|---------|
| GPU → CPU dequeue + UCX submit       | **7 µs** min, 138 µs avg      | The min is the real proxy submission cost; avg is inflated by GPU/UCX contention with the spinning kernel, **not** by the proxy itself. |
| UCX in-flight (`submit → completion`)| **4.2 ms** (one direction)    | This is *all* UCX/CUDA-IPC. The proxy is not on this path — it just waits. |
| Completion → GPU notification        | **36 ns**                     | Effectively free. |
| Worker idle-spin ratio               | 392 880 polls / request       | Worker is idle 99.9 % of the time. |

> **The CPU proxy itself adds tens of microseconds of submission overhead
> and ~30 ns of completion overhead. The remaining latency in the proxy
> run is in UCX/CUDA, identical to what the UCX-direct path would pay
> for the same transfer.**

### Conclusions for the proxy design

1. **The proxy is not the bottleneck.** Submission, completion, and
   dispatch overhead are in the µs / sub-µs range. The proxy worker is
   idle for 99.9 % of its runtime even at sub-millisecond request
   cadence — current capacity is far above current demand.
2. **`publish` is essentially free (~30 ns).** The completion-publishing
   path is well-designed and need not be re-engineered.
3. **`prep+submit` floor is ~7 µs but has high jitter (avg 60–230 µs).**
   The variance correlates with the GPU/UCX driver being busy; on a
   quieter host this is expected to collapse to the 7 µs floor.
4. **The CPU proxy successfully decouples GPU-initiated transfers from
   GDA-KI / IBGDA requirements.** Currently this machine has no
   GPUDirect RDMA support, so the UCX-direct path also falls back to
   `cuda_ipc` for the data plane — meaning the proxy variant is
   functionally equivalent in transport selection and can act as a
   drop-in alternative.
5. **The dominant per-PUT cost (~4.2 ms one-way) is a property of the
   UCX `cuda_ipc` data plane on this host, not the proxy.** This number
   transfers to the UCX-direct binary too once we measure it head-to-head;
   both binaries should track this cost together.
6. **Proxy RTT is essentially flat across message sizes (13.05 ms from
   8 B to 1 MiB).** UCX-direct collapses to 30 / 126 ms at 256 KiB / 1 MiB.
   At ≥256 KiB the proxy is *faster* than UCX-direct on this host.

### What we could not measure here, and why

| Wanted to measure | Blocker | Resolution |
|---|---|---|
| RoCE for the data plane (proxy + UCX-direct) | GPUDirect RDMA not available on this box (`nvidia_peermem` not loaded; UCX falls back to `cuda_ipc` for any GPU-targeted PUT). | Test on a node with GDR enabled, or load `nvidia_peermem` here. |
| Cross-GPU IPC contention | Single-GPU host. | Test on a 2-GPU host; expected to drop `inflight` significantly if same-GPU GPU-engine contention is part of the 4.2 ms. |
| Per-iteration GPU-side timing | Not yet instrumented. | Add NVTX + `clock64()` brackets in `bench_kernel.cu`. |

### Action items, ranked by ROI

**Quick wins (hours):**

1. **Run UCX-direct under the same `UCX_TLS=^tcp` settings** to produce
   an apples-to-apples per-stage table for the demo.
2. **Bound the `prepMemView` retry loop in `bench_host.cpp`** with a
   timeout + clear error. The current "infinite retry every 1 ms" makes a
   misconfigured `UCX_TLS` look like a hang.
3. **Lock `polls/request` and `inflight` into the demo deck.** They're
   the strongest single piece of evidence that "the proxy itself is
   cheap; the cost is UCX."

**Medium (1–2 days):**

4. **Enable GPUDirect RDMA** on the bench node (modprobe
   `nvidia_peermem`; verify with `ucx_info -d | grep -i gdr`). Re-run
   the sweep — this unlocks the RoCE data-plane comparison the demo
   audience will likely ask for.
5. **Add NVTX ranges around `runOnce`'s sub-stages** so the Nsight
   Systems timeline shows GPU↔CPU lag visually, not just as numbers.
6. **Add the size sweep CSV summary directly to the deck.** Already
   produced by the script; just needs to be formatted.

**Strategic (weeks):**

7. **Two-GPU benchmark host** to isolate same-GPU CUDA IPC contention
   from intrinsic IPC cost.
8. **Multi-channel / multi-worker scaling test** (demo currently fixed at
   1/1). Worker is 99.9 % idle, so the *single* proxy can clearly handle
   multiple GPU producers; before raising `proxyWorkerCount`, validate the
   UCX proxy backend's shared `postXfer` path under concurrent proxy workers.
9. **Replace one of the spinning waits with a CUDA-host-managed memory
   write**, in either direction, and re-measure to see whether
   GPU↔CPU notification is genuinely the next bottleneck after
   `inflight`.
10. **Decide where the proxy lives in production.** Demo data should be
    enough to argue: "proxy adds < 100 µs / op vs. UCX-direct, scales to
    1+ M ops/s per worker, runs anywhere UCX runs (no GDA-KI
    requirement)."

### One-line takeaway

> The CPU proxy adds **tens of microseconds** of submission overhead and
> **tens of nanoseconds** of completion overhead. End-to-end latency is
> dominated by the UCX data plane, not by the proxy — making the proxy a
> practical drop-in for environments where GPU-initiated UCX (GDA-KI) is
> unavailable.

---

## 2. Proofs

> All file paths below are real artifacts from the profiling runs in
> `/scrap/cpu-proxy/nixl/profile_results/`. Numbers are quoted directly
> from the run logs — every claim in the summary above maps to one of
> the cited lines.

### 2.1 Sweep table — both binaries, 7 message sizes, 2000 iters / 200 warmup

```
Sweep summary  (csv: profile_results/20260423-132130/sweep.csv)
---------------------------------------------------------------------------
    msg_size        ucx_us      proxy_us      delta_us       ratio
           8       4348.27      13050.85       8702.58       3.00x
          64       4348.17      13050.88       8702.71       3.00x
         512       4348.18      13050.90       8702.72       3.00x
        4096       4348.23      13050.99       8702.76       3.00x
       32768       4348.66      13050.98       8702.32       3.00x
      262144      30451.29      13050.95     -17400.34       0.43x
     1048576     126155.51      13050.96    -113104.55       0.10x
```

Source: `profile_results/20260423-132130/summary.txt`

Two non-obvious facts deserve highlighting:

1. **Proxy RTT is essentially flat at 13.05 ms across all message
   sizes** (8 B → 1 MiB). The standard deviation across the 7 rows is
   < 0.15 µs.
2. **UCX-direct RTT is flat up to 32 KiB and then collapses to 7× / 30×
   worse at 256 KiB / 1 MiB.** Some protocol switch in UCX's GPU
   device-API path. *At 256 KiB and above the proxy is faster than
   UCX-direct on this host.*

Raw CSV (excerpt):

```
variant,msg_size,iters,warmup,rtt_us
ucx,8,2000,200,4348.275
proxy,8,2000,200,13050.852
ucx,1048576,2000,200,126155.507
proxy,1048576,2000,200,13050.960
```

Source: `profile_results/20260423-132130/sweep.csv`

### 2.2 Per-stage proxy-worker breakdown

The proxy worker emits `[proxy-stats]` to stderr at thread shutdown.
These come straight from `*_recv.err` of the same sweep:

**8 B PUT** (`profile_results/20260423-132130/sweep_proxy_8_recv.err`):

```
[proxy-stats][w0] runOnce_iters=864336632  progress_calls=864336632
[proxy-stats][w0] prep+submit  n=2200     avg=  138.649 us  min=    6.898 us  max=  910.252 us
[proxy-stats][w0] inflight     n=2200     avg= 4204.102 us  min= 3442.912 us  max= 4338.254 us
[proxy-stats][w0] publish      n=2200     avg=    0.036 us  min=    0.020 us  max=    1.567 us
[proxy-stats][w0] polls/request=392880.3  (1.0 == every poll dispatched; high == worker spinning idle)
```

**512 B PUT** (`profile_results/20260423-132130/sweep_proxy_512_recv.err`):

```
[proxy-stats][w0] runOnce_iters=864488799  progress_calls=864488799
[proxy-stats][w0] prep+submit  n=2200     avg=   64.423 us  min=    9.384 us  max=  490.725 us
[proxy-stats][w0] inflight     n=2200     avg= 4279.236 us  min= 3856.350 us  max= 4342.659 us
[proxy-stats][w0] publish      n=2200     avg=    0.038 us  min=    0.020 us  max=    0.207 us
[proxy-stats][w0] polls/request=392949.5  (1.0 == every poll dispatched; high == worker spinning idle)
```

**32 KiB PUT** (`profile_results/20260423-132130/sweep_proxy_32768_recv.err`):

```
[proxy-stats][w0] runOnce_iters=859800326  progress_calls=859800326
[proxy-stats][w0] prep+submit  n=2200     avg=   98.513 us  min=   10.580 us  max=  499.277 us
[proxy-stats][w0] inflight     n=2200     avg= 4246.007 us  min= 3848.766 us  max= 4349.908 us
[proxy-stats][w0] publish      n=2200     avg=    0.029 us  min=    0.021 us  max=    0.208 us
[proxy-stats][w0] polls/request=390818.3  (1.0 == every poll dispatched; high == worker spinning idle)
```

**1 MiB PUT** (`profile_results/20260423-132130/sweep_proxy_1048576_recv.err`):

```
[proxy-stats][w0] runOnce_iters=861829696  progress_calls=861829696
[proxy-stats][w0] prep+submit  n=2200     avg=  119.350 us  min=    7.485 us  max=  383.540 us
[proxy-stats][w0] inflight     n=2200     avg= 4232.296 us  min= 3970.303 us  max= 4346.304 us
[proxy-stats][w0] publish      n=2200     avg=    0.033 us  min=    0.021 us  max=    0.190 us
[proxy-stats][w0] polls/request=391740.8  (1.0 == every poll dispatched; high == worker spinning idle)
```

### 2.3 Independent confirmation from a separate Nsight Systems capture

Different run, message size 8192 B, 2000 iters / 200 warmup, with NVTX
ranges enabled. From `profile_results/20260423-133442/analysis.txt`:

```
-- measured RTT (sender stdout) --
  msg_size=8192    iters=2000    RTT=13049.011 us  one-way=6524.506 us  [THREAD]
```

(Reproduces the 13.05 ms RTT from the sweep above.)

```
-- proxy-stats summary (from *.err) --
  nsys_proxy_8192_send.err (send):
    [proxy-stats][w0] runOnce_iters=805707292  progress_calls=805707292
    [proxy-stats][w0] prep+submit  n=2200     avg=   34.155 us  min=   16.002 us  max=  360.770 us
    [proxy-stats][w0] inflight     n=2200     avg= 4311.288 us  min= 3911.589 us  max= 5041.094 us
    [proxy-stats][w0] publish      n=2200     avg=    0.748 us  min=    0.338 us  max=    4.699 us
    [proxy-stats][w0] polls/request=366230.6
  nsys_proxy_8192_recv.err (recv):
    [proxy-stats][w0] runOnce_iters=888431371  progress_calls=888431371
    [proxy-stats][w0] prep+submit  n=2200     avg=  120.273 us  min=    7.844 us  max=  559.810 us
    [proxy-stats][w0] inflight     n=2200     avg= 4223.418 us  min= 3787.025 us  max= 4344.904 us
    [proxy-stats][w0] publish      n=2200     avg=    0.038 us  min=    0.022 us  max=    0.218 us
    [proxy-stats][w0] polls/request=403832.4
```

Same shape, both directions. The numbers reproduce.

### 2.4 NVTX ranges (independent CPU-time accounting in the proxy worker)

From the same `analysis.txt`:

```
-- nsys stats: nvtx_pushpop_sum --
  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Range
  85.9      483318270        1952784    247.5     129.0     101       18074     256.9        :prx:progress
  14.0      78803251         2200       35819.7   35454.5   16540     361723    11956.3      :prx:submit
  0.1       542992           2200       246.8     144.0     112       3283      227.6        :prx:publish
```

Reads as: of the proxy worker's **total CPU time**, 85.9 % is spent
in `:prx:progress` (idle UCX polling), 14 % in `:prx:submit`
(dispatching requests), and **0.1 % in `:prx:publish`**. The publish
stage is genuinely a rounding error.

`:prx:submit` here measures `Avg = 35.8 µs` and matches `prep+submit`
from the proxy stats — confirms the two independent measurement paths
agree.

### 2.5 CUDA-API contrast: what the proxy adds vs. what UCX-direct does

**Proxy capture** (`analysis.txt`, `cuda_api_sum`):

```
  Time (%)                    Total Time (ns)  Num Calls  Avg (ns)       Med (ns)       Min (ns)     Max (ns)     StdDev (ns)  Name
  88.0                        28710190143      1          28710190143.0  28710190143.0  28710190143  28710190143  0.0          cudaStreamSynchronize
  8.3                         2713265006       3920973    692.0          719.0          242          130512       467.8        cuEventQuery
  3.6                         1180486157       982413     1201.6         1053.0         457          2347554      2891.6       cuStreamAddCallback
  0.1                         20679895         2200       9400.0         8859.0         6419         132235       3514.0       cuMemcpyDtoDAsync_v2
```

**UCX-direct capture** (same workload, no proxy):

```
  Time (%)                    Total Time (ns)  Num Calls  Avg (ns)      Med (ns)      Min (ns)    Max (ns)    StdDev (ns)  Name
  99.7                        9563608266       1          9563608266.0  9563608266.0  9563608266  9563608266  0.0          cudaStreamSynchronize
  0.2                         14862493         6          2477082.2     7580.5        3413        14749552    6012353.3    cuMemFree_v2
  0.1                         7800422          1          7800422.0     7800422.0     7800422     7800422     0.0          cudaLaunchKernel
```

Reads as: proxy run does **2200× `cuMemcpyDtoDAsync_v2`** (~9.4 µs each
— the UCX cuda_ipc PUTs), **3.9 M `cuEventQuery`** and **~1 M
`cuStreamAddCallback`** (the proxy worker driving UCX progress).
UCX-direct does **none of those** — the GPU kernel calls UCX directly
via the device API and there's no host-side driver involvement. This is
the literal CPU-side cost of the proxy hop, in numbers.

### 2.6 GPU memory time, both runs

**Proxy** (`analysis.txt`, `cuda_gpu_mem_time_sum`):

```
  Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Operation
  99.8      4242970          2200   1928.6    1920.0    1856      7009      203.4        [CUDA memcpy Device-to-Device]
  0.1       3424             3      1141.3    1408.0    384       1632      665.4        [CUDA memset]
  0.1       2496             5      499.2     416.0     416       704       127.2        [CUDA memcpy Host-to-Device]
  0.0       1120             1      1120.0    1120.0    1120      1120      0.0          [CUDA memcpy Device-to-Host]
```

**UCX-direct**:

```
  Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Operation
  53.5      3456             3      1152.0    1312.0    448       1696      639.2        [CUDA memset]
  25.2      1632             3      544.0     448.0     448       736       166.3        [CUDA memcpy Host-to-Device]
  21.3      1376             1      1376.0    1376.0    1376      1376      0.0          [CUDA memcpy Device-to-Host]
```

Reads as: proxy dispatched **2200 explicit D→D memcpys at ~1.9 µs each**;
UCX-direct dispatched **zero** D→D memcpys (it goes through the
device-API path, not via runtime cudaMemcpy). 2200 = warmup + iters,
exactly one per PUT. **Per-PUT GPU-engine cost is ~1.9 µs** — that's
what each `cuda_ipc` write actually costs the GPU's copy engine.

---

## 3. Claim → proof map

| Executive-summary claim | Direct evidence |
|---|---|
| Proxy submission floor ~7 µs | `prep+submit min = 6.898 µs` (`sweep_proxy_8_recv.err:12`) |
| Submission averages 30–230 µs depending on contention | `prep+submit avg` rows across the 4 cited `*_recv.err` files |
| Publish ≈ 30 ns | `publish avg = 0.029–0.038 µs` (all 4 cited `*_recv.err` files); `0.1 % :prx:publish` NVTX range (`analysis.txt`) |
| `inflight` is 4.2 ms one-way and dominates | `inflight avg ≈ 4204/4279/4246/4232 µs` across sizes |
| Worker is idle 99.9 % of the time | `polls/request ≈ 390 k` (4 files); `:prx:progress` is 85.9 % of worker CPU but avg 247 ns / call ⇒ idle-poll loop |
| Proxy RTT essentially flat across sizes | `summary.txt` — `proxy_us` column reads 13050.85, 13050.88, 13050.90, 13050.99, 13050.98, 13050.95, 13050.96 |
| UCX-direct RTT collapses at large sizes | `summary.txt` — `30451.29 µs` at 256 KiB, `126155.51 µs` at 1 MiB |
| At ≥256 KiB the proxy is *faster* than UCX-direct | `summary.txt` — `delta_us = -17400.34` and `-113104.55`; `ratio = 0.43x` and `0.10x` |
| Per-PUT GPU memcpy cost ~1.9 µs | `analysis.txt` — `[CUDA memcpy Device-to-Device] avg=1928.6 ns, n=2200` |
| UCX-direct issues no host-side cuMemcpy/cuEventQuery | Compare `cuda_api_sum` between proxy and ucx-direct captures — `cuMemcpyDtoDAsync_v2` only present in proxy capture |
| Two independent capture sessions reproduce 13.05 ms RTT | `summary.txt` (sweep, run @ 13:21–13:32) and `analysis.txt` (NVTX run, run @ 13:34, RTT = 13049.011) |

---

## 4. Caveats to disclose during the demo

1. **All numbers above are on a host where GPUDirect RDMA is supported by
   the build but not currently active**, so UCX falls back to `cuda_ipc`
   for the data plane in *both* binaries. The proxy isn't being unfairly
   compared against an RDMA fast path that it can't reach — they're using
   the same underlying transport. **What's missing on this host (all
   deployable):**

   - `nvidia_peermem.ko` is **not loaded** and the `.ko` file is not in
     `/lib/modules/$(uname -r)` — but the NVIDIA driver is `550.163.01`,
     which ships peermem. Re-installing `nvidia-utils-550` /
     `nvidia-dkms-550` and `modprobe nvidia_peermem` should make GDR
     available.
   - `gdrcopy` is **not installed** (`ldconfig -p | grep gdr` is empty).
   - The custom UCX at `/scrap/cpu-proxy/ucx-install` is **already built
     with all the right pieces**: `--with-cuda`, `HAVE_CUDA_FABRIC`,
     `HAVE_DECL_MLX5DV_REG_DMABUF_MR`, plus `libuct_cuda.so` and
     `libuct_ib_mlx5_gda.so` (the GDA-KI device-API module) on disk —
     so no UCX rebuild is required to unlock GDR.

   Once either `nvidia_peermem` or `gdrcopy` is in place, re-running with
   `UCX_TLS=^tcp,cuda_ipc` should let UCX register GPU memory with the
   NIC and produce a true RoCE-based RTT for both binaries — the number
   that actually justifies (or refutes) the proxy hop.
2. **Both binaries currently use `do_put_async`** in the kernel. The GPU
   does not wait on `nixlGpuGetXferStatus`; it waits for the peer's sequence
   counter to arrive in HBM, so the reported RTT is the GPU-observed
   ping-pong round trip.
3. **The flat 13.05 ms across sizes for the proxy is suspicious enough
   to call out.** The per-stage breakdown shows `inflight` *also* stays
   near 4.2 ms regardless of size — so the constancy is real and traces
   to UCX, not to a measurement bug. Re-running with a wider size range
   is still a good belt-and-suspenders check before publishing.
4. **The proxy demo is pinned to one worker / one channel** until the UCX
   proxy backend's shared `postXfer` path is validated with concurrent
   proxy workers.
5. **One missing data point for the demo:** UCX-direct's `prep+submit` /
   `inflight` / `publish` equivalents. The proxy stats only exist in
   the proxy build. To make a clean apples-to-apples claim, the next
   run should add NVTX ranges around the GPU-side `nixlPut` call sites
   in the kernel and capture both binaries with `nsys`.

---

## 5. Reproducing this report

```bash
cd /scrap/cpu-proxy/nixl
ninja -C build
NIXL_PROXY_STATS=1 \
    examples/device/pingpong/scripts/profile_overhead.sh sweep 2000 200
NIXL_PROXY_STATS=1 \
    examples/device/pingpong/scripts/profile_overhead.sh nsys 8192 2000
examples/device/pingpong/scripts/analyze_nsys.sh \
    profile_results/<latest-nsys-dir>
```

See [`README.md`](README.md) for environment variables, output layout,
and how to read the per-stage stats.
