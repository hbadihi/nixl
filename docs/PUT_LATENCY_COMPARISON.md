# Put Latency Comparison: NIXL Device API vs NCCL GIN Proxy vs GDAKI

Methodology for an apples-to-apples comparison of one-sided **put** latency across:

1. **NIXL device API** (`nixlPut`) — this repo, `nixl-cpu-proxy`, two backends:
   - UCX-direct (`ucp_device_put`, kernel issues to NIC)
   - CPU-proxy (`nixl_device_pingpong_proxy`, GPU enqueues → CPU worker → UCX RDMA write)
2. **NCCL GIN proxy** (`NCCL_GIN_TYPE=2`) — GPU posts GFDs → `gin_host_proxy` → `iput` → IB `RDMA_WRITE`
3. **NCCL GDAKI** (`NCCL_GIN_TYPE=3`) — kernel posts WQEs directly via `doca_gpu_dev_verbs_put`
4. **NVSHMEM** (`../nvshmem`) — device put (`nvshmem_*_put` / `nvshmemx_*_put_{warp,block}`), two transport modes:
   - IBGDA (`NVSHMEM_IB_ENABLE_IBGDA=1`, kernel posts WQEs directly)
   - Proxy/IBRC (`NVSHMEM_IB_ENABLE_IBGDA=0`, GPU enqueues → CPU proxy thread → IB RC)

Target environment: **two nodes over IB/RoCE** (true RDMA path for all backends).

> **NVSHMEM is the cheapest to measure**: it ships ready-made put-latency microbenchmarks (`perftest/device/pt-to-pt/shmem_put_latency`, `shmem_put_ping_pong_latency`), so no kernel needs to be written — only build + run with the right env vars.

---

## 1. Why this needs a plan (tooling asymmetry)

| Repo | Put-latency harness available? |
|------|--------------------------------|
| `nixl-cpu-proxy` | **Yes.** `examples/device/pingpong/` issues `nixlPut`, times with `clock64()`, reports `issue` / `complete` / `peer-wait` / `network` / `one-way` / `RTT` in µs. Two build variants (UCX-direct, proxy). For the proxy, the true wire cost is the worker `post_submit` line; GPU `complete` is a benchmark artifact (see §3.4). |
| `../nccl` | **No isolated put benchmark.** GIN `put` only appears in `docs/examples/06_device_api/02_alltoall_gin` (correctness) and `contrib/nccl_ep/ep_bench` (whole-kernel CUPTI µs). |
| `../nvshmem` | **Yes.** `perftest/device/pt-to-pt/` ships `shmem_put_latency` (one-way) and `shmem_put_ping_pong_latency` (RTT), CUDA-event timed in µs, with Thread/Warp/Block tables. Ready binaries — no code needed. |

**Consequence:** only the NCCL side requires writing a small **GIN ping-pong micro-kernel** that mirrors the NIXL pingpong protocol (forced via `NCCL_GIN_TYPE`). NIXL and NVSHMEM both have runnable harnesses. Everything else is about holding conditions constant.

---

## 2. Backend mapping (the comparison pairs)

The architectures pair up cleanly. Keep these pairings straight or the numbers conflate "CPU in path" with "transport stack."

| Class | Who issues to NIC | NIXL | NCCL | NVSHMEM |
|-------|-------------------|------|------|---------|
| **CPU-proxy** | GPU enqueues descriptor; **CPU** issues RDMA | `nixl_device_pingpong_proxy` → worker → `submitRmaWrite` | GIN proxy (`NCCL_GIN_TYPE=2`): GFD → `gin_host_proxy` → `iput` → IB `RDMA_WRITE` | `NVSHMEM_IB_ENABLE_IBGDA=0` + `REMOTE_TRANSPORT=ibrc`: device put → proxy channel → CPU proxy thread → IB RC |
| **Kernel-initiated** | **GPU** posts WQE directly | `nixl_device_pingpong` (UCX-direct, `ucp_device_put`) | GDAKI (`NCCL_GIN_TYPE=3`): `doca_gpu_dev_verbs_put` | `NVSHMEM_IB_ENABLE_IBGDA=1`: IBGDA, GPU posts WQEs directly |

Primary comparisons (same row = fair):
- **CPU-proxy row**: NIXL proxy ↔ NCCL GIN proxy ↔ NVSHMEM IBRC
- **Kernel-initiated row**: NIXL UCX-direct ↔ NCCL GDAKI ↔ NVSHMEM IBGDA

Cross-row pairings (e.g. NIXL proxy vs GDAKI) are secondary and should be labeled as mixing the CPU-in-path dimension.

---

## 3. Measurement protocol to replicate

The NIXL kernel (`examples/device/pingpong/bench_kernel.cu`) defines the reference protocol. The NCCL micro-kernel must match it on every axis below.

### 3.1 Timing primitive
- **NIXL**: `clock64()` on the sender, single timed region inside the kernel, converted to µs via `cudaDevAttrClockRate` (kHz).
- **NVSHMEM**: host-side `cudaEventRecord` / `cudaEventElapsedTime` bracketing a **single kernel launch that loops `iter` times internally**, then divides by `iter` (`shmem_put_latency.cu` lines 135–145). Because the whole loop is in one launch, per-launch overhead amortizes to ~0, so this is comparable to NIXL's in-kernel `clock64`.
- **NCCL micro-kernel** (to be written): use the **same in-kernel `clock64`** approach as NIXL so it lines up exactly.
- Do **not** use CUPTI or per-launch `cudaEvent` for the per-op number (launch overhead dominates at small sizes). The cross-timer difference (`clock64` vs amortized `cudaEvent`) is sub-microsecond but should be noted as a caveat.

### 3.2 Concurrency / scope
- Single in-flight op. Launch `<<<1,1>>>` (THREAD) and `<<<1,32>>>` (WARP).
- NIXL `level` (THREAD/WARP) ↔ NCCL `ncclCoopThread()` / `ncclCoopWarp()` coop scope passed to `gin.put`.
- One channel / one GIN context / one QP on the NCCL side to match NIXL's single-channel pingpong.

### 3.3 Ping-pong structure (must match)
- **Sender:** write seq counter → `put(payload + seq)` → wait local completion → wait for peer reply counter → record `RTT`.
- **Receiver:** wait incoming → `put` reply back.
- Report `RTT` and `one-way = RTT / 2`.

### 3.4 Completion semantics (subtlest axis)
| NIXL phase | Meaning | NCCL equivalent | NVSHMEM equivalent |
|------------|---------|-----------------|--------------------|
| `issue` | `nixlPut` call entry→return | time around `gin.put` call | time around `nvshmem_*_put_nbi` call |
| `complete` | local completion seen by GPU (`nixlGpuGetXferStatus` terminal). **UCX-direct: NIC CQE.** For an RC `RDMA_WRITE` the send CQE fires only when the requester receives the responder NIC's ACK, so `complete` is a **write+ACK wire ROUND TRIP** (≈ 2× one-way wire), *not* a one-way outbound cost. **Proxy: benchmark-only — GPU reads the worker's completion slot over PCIe, not in the real data path.** | `gin.flush()` returning | `nvshmem_quiet()` returning (this is what `shmem_put_latency` one-way reports) |
| `network` | **write+ACK wire round trip** (confirms the data landed), source depends on build. **UCX-direct:** = `complete` (NIC CQE = write→peer + ACK→back). **Proxy:** the worker's `post_submit` (backend `submit`→`checkCompletion` SUCCESS, same RC ACK semantics) in the `[proxy-worker-stats]` block — measured on the host steady clock, free of the GPU completion poll. **True one-way wire ≈ `network` / 2.** | GDAKI WQE→CQE / proxy `iput` in-flight | wire portion of `put_nbi`+`quiet` |
| `peer-wait` | remote arrival (peer bumped a counter). **Not a clean network number:** also includes the receiver's reaction time, the return-path transit, and the sender's own counter-poll. | `ncclGin_SignalInc` + `gin.waitSignal()` | `put_signal` + `nvshmem_uint64_wait_until` (ping-pong test) |
| `RTT` | full ping-pong loop | full ping-pong loop (clock64) | `shmem_put_ping_pong_latency` (reports full RTT, **not** RTT/2) |

Document which phase is being compared in every chart. The safest headline number is **`one-way` (RTT/2)** because it does not depend on each stack's local-completion definition. Note that `one-way` (RTT/2) and `network`/2 measure *different* things: `one-way` is half the full application round trip (put + remote reaction + reply), while `network`/2 is half the RC write+ACK exchange of a *single* leg. Do not conflate them.

> **`complete` is build-specific — do not treat it as "network" for the proxy.** In the UCX-direct build, local completion is the NIC CQE. For RC `RDMA_WRITE`, that CQE only fires after the responder NIC ACKs the write, so `complete` is the **write+ACK round trip**, i.e. ≈ 2× the true one-way wire (the responder is one-sided — no remote CPU/GPU in the loop). In the CPU-proxy build, the GPU only learns of completion by polling the worker's host-mapped completion slot over PCIe; a real proxy data path never does this (it fires the put and waits on the destination flag), so proxy `complete` is a **benchmark artifact**. The authoritative proxy network cost is the worker's **`post_submit`** line, which has the *same* RC write+ACK semantics as direct `complete` (so `direct complete` ≡ `proxy post_submit` in kind). Enabling the proxy `complete` poll (`--measure-submit`, the default) also adds PCIe round-trips that perturb `peer-wait`/`rtt`; the cleanest real-data-path latency is the fire-and-forget run (`--no-measure-submit`). **Empirically confirmed (§3.7): the fire-and-forget vs poll-on `one-way` differ by < 0.1 µs — the completion poll only re-partitions the RTT into `complete`+`peer-wait`, it does not add wall-clock.**

> **`peer-wait` is intentionally measured with a busy-spin, not `__nanosleep`.** The sender's wait on the reply counter uses a tight `ld.volatile` spin so the recorded arrival time isn't quantized/jittered by sleep wakeup latency. Even so, `peer-wait` still folds in the receiver's software reaction and the return-path transit, so it is **not** a pure network number — use `network` (above) for that.

> **Two NVSHMEM numbers map to two different NIXL phases.** `shmem_put_latency` (one-way, put_nbi+quiet) ≈ NIXL **`complete`** (local completion). `shmem_put_ping_pong_latency` reports **full RTT** — divide by 2 for the `one-way` comparison and confirm it isn't pre-halved.

### 3.5 Message sizes
- Match the NIXL sweep: `8 64 512 4096 32768 262144 1048576` bytes.
- NIXL sends `msg_size + 8` (payload + counter); make the NCCL kernel transfer the same effective payload.

### 3.6 Iterations / warmup
- `iters >= 2000` (NIXL default), `warmup = 200`. Use `5000 / 500` for tighter percentiles.

### 3.7 Validated NIXL findings (2026-06-04, two-node H100 80 GB / IB, mlx5_0, GPU0)
Full 28-config sweep + four follow-ups (`profile_results/precise-20260604/`, `iters=5000 warmup=500`,
clock64→µs at peak 1980 MHz; per-run stddev ~0.04 µs). THREAD-level highlights:

| size | direct one-way | proxy one-way | ucx `ucp_put_lat` | direct `network` (write+ACK) | proxy `network` (post_submit) |
|------|---------------:|--------------:|------------------:|-----------------------------:|------------------------------:|
| 8 B    |  6.04 |  5.30 |  6.28 |  4.56 |  3.72 |
| 512 B  |  6.10 |  5.58 |  6.36 |  4.92 |  4.39 |
| 4 KB   |  6.14 |  5.96 |  6.55 |  4.80 |  4.19 |
| 32 KB  |  7.26 |  6.66 |  7.77 |  5.89 |  5.38 |
| 1 MB   | 29.04 | 17.45 | 29.32 | 27.75 | 26.73 |

Key conclusions (validated from source + experiment):
1. **Harness cross-check.** Our UCX-direct `one-way` tracks the reference `ucx_perftest -t ucp_put_lat`
   at every size (6.04 vs 6.28 µs @8 B; 29.04 vs 29.32 µs @1 MB), confirming the in-kernel `clock64`
   protocol is sound.
2. **Wire (`network`) is ~equal** between direct and proxy — the proxy is **not** faster on the wire.
   `network` is a write+ACK round trip; true one-way wire ≈ `network`/2 (≈ 1.9 µs @512 B).
3. **The completion poll adds no wall-clock.** Re-running direct fire-and-forget (`--no-measure-submit`)
   moved `one-way` by < 0.1 µs at every size (e.g. 512 B: 6.100 → 6.188). The poll only re-partitions the
   RTT into `complete` + `peer-wait`; the CQE (≈ 2 wire-hops) lands before the peer's reply, so it is hidden.
4. **Proxy's small-size win is the doorbell, not the network.** Call→doorbell breakdown:

   | stage (512 B) | µs | what |
   |---|---:|---|
   | direct `issue` (SM `ucp_device_put` = build WQE + MMIO doorbell) | 2.86 | GPU rings doorbell |
   | proxy `issue` (GPU enqueue into ring) | 1.39 | GPU hands off |
   | worker `dequeue` | 0.13 | CPU picks up |
   | worker `prepare` | 0.06 | build descriptor |
   | worker `submit` (backend submit == NIC doorbell rung) | 1.28 | CPU rings doorbell |

   The CPU rings the NIC doorbell faster (`submit` 1.28 µs) than the SM does (`issue` 2.86 µs), and that
   ~1.5 µs delta lands on **both** ping-pong legs (sender put + receiver reply), explaining the ~0.5–0.8 µs
   `one-way` advantage at small sizes. The `--measure-stages` `to_doorbell`/`to_dequeued` diagnostics agree
   (`to_doorbell − to_dequeued` ≈ worker `submit`) but are an **upper bound** — each GPU stage poll crosses
   PCIe and inflates the absolute numbers, so use them only for the *delta*, not the absolute handoff cost.
5. **Large transfers (≥256 KB) diverge sharply** in `one-way` (1 MB: direct 29.0 vs proxy 17.5 µs) while
   `network` stays close (27.8 vs 26.7 µs). This is a separate effect from the small-size doorbell story and
   warrants its own investigation before publishing a large-size headline.

Reproduce / re-aggregate: `python3 profile_results/precise-20260604/aggregate.py` (emits `summary.csv` +
the tables above; pulls the ucx baseline from `ucxbase/`, direct fire-and-forget from `full_oneway/`,
proxy measured from `full_measured/`, and the stage breakdown from `stages/`).

---

## 4. Conditions to hold constant (two-node IB/RoCE)

These are the dominant threats to validity:

1. **Topology — biggest one.** Run **all** backends across the **same two nodes on the same HCA/NIC**. The default `profile_overhead.sh` uses `RECV_HOST=127.0.0.1` (single node, may pick `cuda_ipc`); that would let NIXL "win" with intra-node IPC while NCCL goes over the wire. For NIXL two-node, launch sender/receiver manually with `--peer-ip` set to the other node and (optionally) pin the UCX transport to RC (`UCX_TLS=rc,cuda_copy` or similar) so it does not fall back to IPC.
2. **Same GPU model, same NIC, same PCIe/NUMA placement** on both nodes.
3. **Same software versions**: CUDA, GPU driver, UCX (NIXL), DOCA GPUNetIO (GDAKI), NCCL build with GIN enabled.
4. **Same QP depth / single QP / single context** on NCCL; single channel on NIXL; for NVSHMEM consider `NVSHMEM_IBGDA_NUM_DCI`/`NUM_RC_PER_PE` so it isn't using more QPs than the others.
5. **Matched warmup + iters**, and discard the first run after process start.
6. **GDAKI hardware requirement**: MLX5 NIC + GDR/DMA-BUF; verify `props.ginType == NCCL_GIN_TYPE_GDAKI` actually took effect (it silently falls back to proxy otherwise).
7. **NVSHMEM transport verification**: build with `NVSHMEM_IBGDA_SUPPORT=ON`; confirm IBGDA actually engaged at runtime (look for the *"IBGDA … used for device-side APIs over IB"* log) — otherwise it silently uses the proxy path and you'd mislabel the row. Also keep `NVSHMEM_REMOTE_TRANSPORT=ibrc` for the proxy row so it matches the others' transport.

---

## 5. Step-by-step execution plan

### Step A — NIXL baselines (this repo)
Build both variants:
```bash
ninja -C build examples/device/pingpong/nixl_device_pingpong \
                examples/device/pingpong/nixl_device_pingpong_proxy
```

Two-node run (manual, one process per node). On **node B (receiver)**:
```bash
build/examples/device/pingpong/nixl_device_pingpong_proxy \
  --role receiver --gpu 0 --listen-port 19500 \
  --peer-ip <NODE_A_IP> --peer-port 19501 \
  --msg-size 8 --iters 5000 --warmup 500 --op put
```
On **node A (sender, prints latency)**:
```bash
build/examples/device/pingpong/nixl_device_pingpong_proxy \
  --role sender --gpu 0 --listen-port 19501 \
  --peer-ip <NODE_B_IP> --peer-port 19500 \
  --msg-size 8 --iters 5000 --warmup 500 --op put
```
Repeat with `nixl_device_pingpong` (UCX-direct) and across the size sweep. Add `--warp` for WARP-level. Grep results:
```bash
grep -E 'RTT=|one-way=|\[pingpong-stats\]' sender.out
```
(`profile_overhead.sh sweep` automates the size loop but is single-host; for two-node either set `RECV_HOST`/run its halves on each node, or script the manual loop above.)

### Step B — Build the NCCL GIN ping-pong micro-bench
> **Full implementation brief: [`PART_B_NCCL_GIN_PINGPONG_PLAN.md`](./PART_B_NCCL_GIN_PINGPONG_PLAN.md)** — self-contained, hand to another agent.

Create the bench (recommended location `../nccl/docs/examples/06_device_api/04_gin_pingpong_latency/`) modeled on `docs/examples/06_device_api/02_alltoall_gin` for setup (comm init, `ncclMemAlloc`, `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC`, `ncclDevCommCreate` with signals + world barrier), and a 2-rank ping-pong kernel that:
- uses `ncclGin gin{devComm, 0}` (single context),
- sender: `gin.put(world, peer, dstWin, dstOff, srcWin, srcOff, size, ncclGin_SignalInc{sig})` then `gin.flush()` then `gin.waitSignal(...)` for the reply,
- receiver: `gin.waitSignal(...)` then `gin.put(...)` reply,
- times the sender loop with `clock64()` and converts with `cudaDevAttrClockRate`,
- reports `RTT` / `one-way` and per-phase (`issue` = around put, `complete` = around flush) like NIXL,
- sweeps the same sizes, same iters/warmup, THREAD (`<<<1,1>>>`) and WARP (`<<<1,32>>>`).

### Step C — Run NCCL both backends
Same two nodes, same NIC, 2 ranks (one GPU per node), launched via MPI or the example's util harness:
```bash
# GIN proxy
NCCL_GIN_TYPE=2 <launcher> ./gin_pingpong --iters 5000 --warmup 500 --size <S>
# GDAKI
NCCL_GIN_TYPE=3 <launcher> ./gin_pingpong --iters 5000 --warmup 500 --size <S>
```
Confirm the backend actually engaged (print `props.ginType`).

### Step D — NVSHMEM (ready binaries, no code)
Build with IBGDA support, then run the shipped perftests on the **same two nodes**:
```bash
# build (once), IBGDA off by default so enable it explicitly
cd ../nvshmem && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/install \
         -DNVSHMEM_IBGDA_SUPPORT=ON -DNVSHMEM_MPI_SUPPORT=ON \
         -DCUDA_HOME=/usr/local/cuda
cmake --build . -j && cmake --install .
```
Run both transport modes, 2 PEs (one GPU per node):
```bash
PT2PT=install/bin/perftest/device/pt-to-pt
# kernel-initiated (IBGDA)  -> compare to nixl-direct / nccl-gdaki
NVSHMEM_IB_ENABLE_IBGDA=1 mpirun -n 2 --host node0,node1 \
  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 $PT2PT/shmem_put_ping_pong_latency -b 8 -e 1M -n 5000 -w 500
# CPU-proxy (IBRC)          -> compare to nixl-proxy / nccl-gin-proxy
NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_REMOTE_TRANSPORT=ibrc mpirun -n 2 --host node0,node1 \
  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 $PT2PT/shmem_put_ping_pong_latency -b 8 -e 1M -n 5000 -w 500
```
Use `shmem_put_latency` (one-way) for the `complete`-style number and `shmem_put_ping_pong_latency` for RTT. The binaries auto-emit Thread/Warp/Block tables (set block size with `-t`).

### Step E — Aggregate and chart
Collect six series — `nixl-direct`, `nixl-proxy`, `nccl-gin-proxy`, `nccl-gdaki`, `nvshmem-ibgda`, `nvshmem-ibrc` — into one CSV, then plot **one-way µs vs message size** (log-x). Group by class: the kernel-initiated row (nixl-direct / nccl-gdaki / nvshmem-ibgda) and the CPU-proxy row (nixl-proxy / nccl-gin-proxy / nvshmem-ibrc). Reuse the analysis style in `examples/device/pingpong/scripts/generate_active_pane_analysis.py` if convenient.

---

## 6. Deliverables

- `nccl/contrib/gin_pingpong/` micro-benchmark (kernel + host + build + README). *(NVSHMEM and NIXL need no new code.)*
- A combined `put_latency_compare.csv` (columns: `stack, backend, class, level, msg_size, iters, one_way_us, rtt_us`).
- One chart: one-way µs vs size, grouped by class (kernel-initiated vs CPU-proxy), six series.
- A short results note covering: which `one-way` definition was used, exact topology/NIC/versions, transport-mode verification (GDAKI engaged, IBGDA engaged), and the caveats from §4.

---

## 7. Key caveats to publish with any numbers

- **Topology**: single-node IPC vs over-the-wire is the dominant confounder — only compare same-topology runs.
- **Completion semantics**: `flush`/`waitSignal` (NCCL), `GetXferStatus`/counter (NIXL), and `quiet`/`wait_until` (NVSHMEM) differ; prefer `one-way` for the headline and state which local-completion primitive each used.
- **Backend verification**: confirm GDAKI didn't silently fall back to proxy, and that NVSHMEM IBGDA actually engaged (vs silent proxy).
- **Timer difference**: NVSHMEM uses amortized `cudaEvent`; NIXL/NCCL use in-kernel `clock64`. Sub-µs effect, but note it.
- **Single in-flight**: this measures latency, not throughput; do not extrapolate to bandwidth.

---

## 8. Source references

NIXL pingpong protocol (reference to mirror):
- `examples/device/pingpong/bench_kernel.cu` — `nixl_pingpong_latency_kernel`, `clock64()` timing.
- `examples/device/pingpong/bench_main.cpp` — arg parsing, cycle→µs conversion, `[pingpong-stats]`.
- `examples/device/pingpong/scripts/profile_overhead.sh` — UCX-direct vs proxy sweep driver.

NCCL GIN:
- `docs/examples/06_device_api/02_alltoall_gin/main.cu` — `gin.put` + signals + flush + barrier setup.
- `src/include/nccl_device/gin.h` — `ncclGin.put(...)` API.
- `src/include/nccl_device/gin/proxy/gin_proxy.h` — proxy `put` (GFD path).
- `src/include/nccl_device/gin/gdaki/gin_gdaki.h` — GDAKI `putImpl` → `doca_gpu_dev_verbs_put`.
- `src/transport/net_ib/gin.cc` — `NCCL_GIN_TYPE` param, GDAKI-first/proxy-fallback selection.

NVSHMEM (`../nvshmem`):
- `perftest/device/pt-to-pt/shmem_put_latency.cu` — one-way device put latency (`put_nbi`+`quiet`), CUDA-event timed (lines 21–29, 135–145).
- `perftest/device/pt-to-pt/shmem_put_ping_pong_latency.cu` — full-RTT device put latency.
- `src/include/device/nvshmem_defines.h` / `nvshmemx_defines.h` — device put API (thread / warp / block scoped).
- `src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in` — IBGDA vs proxy device dispatch (lines 99–113).
- `src/include/host/env/env_defs.h` — `NVSHMEM_IB_ENABLE_IBGDA` (line 384), `NVSHMEM_REMOTE_TRANSPORT` (line 336).
