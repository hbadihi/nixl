# Part B — Build an NCCL GIN Put-Latency Ping-Pong Micro-Benchmark

> **Handoff doc.** This is a self-contained implementation brief for an agent who does **not** have the surrounding conversation. Read it top to bottom; it contains all context, exact APIs, file layout, code sketches, build/run commands, output format, and acceptance criteria needed to finish the task without further questions.

---

## 0. Context & why this exists

We are comparing one-sided GPU-initiated **put latency** across three stacks on the **same two nodes over IB/RoCE**:

| Stack | Status |
|-------|--------|
| **NIXL** (`nixl-cpu-proxy`) | Has a put-latency ping-pong harness already (`examples/device/pingpong/`). Done. |
| **NVSHMEM** (`../nvshmem`) | Ships ready put-latency perftests (`shmem_put_latency`, `shmem_put_ping_pong_latency`). Done (build + run only). |
| **NCCL GIN** (`../nccl`) | **No isolated put-latency benchmark exists.** ← *This task (Part B).* |

**Your job:** write a 2-rank **GIN ping-pong** micro-benchmark in the NCCL tree that mirrors the NIXL ping-pong protocol exactly, so its numbers are directly comparable. It must support both GIN backends, selected at runtime:

- **GIN proxy** (`NCCL_GIN_TYPE=2`): GPU enqueues GFDs → CPU progress thread → IB `RDMA_WRITE`. (Compare against NIXL CPU-proxy + NVSHMEM IBRC.)
- **GDAKI** (`NCCL_GIN_TYPE=3`): kernel posts WQEs directly via DOCA GPUNetIO. (Compare against NIXL UCX-direct + NVSHMEM IBGDA.)

All paths in this doc are absolute. The NCCL repo root is:
```
/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor/nccl
```

---

## 1. Reference protocol to mirror (NIXL)

The NIXL kernel is the spec. Reproduce its structure and reported metrics. Key facts (do **not** need the NIXL repo to implement, but for fidelity):

- Single in-flight op, **single thread** (`<<<1,1>>>`, THREAD level) or **single warp** (`<<<1,32>>>`, WARP level).
- Timed on the **sender** with in-kernel **`clock64()`**, converted to microseconds on the host using `cudaDevAttrClockRate` (kHz).
- Ping-pong: sender issues a put to the receiver, waits for local completion, then waits for the receiver's reply; receiver waits for the incoming put then replies with its own put.
- Reported metrics (per timed iteration, aggregated avg/p50/p90/p99/min/max/stddev):
  - `issue` — put call entry→return
  - `complete` — local completion (source safe / op done) from the GPU's view
  - `peer-wait` — time until the peer's reply is observed
  - `one-way` — `RTT / 2`
  - `RTT` — full ping-pong round trip
- Sender prints a summary line:
  ```
  op=put          msg_size=8     iters=1000  RTT=12.047 us  one-way=6.023 us  [THREAD]
  [pingpong-stats] issue       n=1000 avg=...
  ...
  ```

Your NCCL benchmark must emit a comparable summary so the Part E aggregator can parse all stacks uniformly.

**Payload size note:** NIXL transfers `msg_size + 8` bytes (payload + an embedded sequence counter). In GIN, the sequence/handshake is carried by **signals**, not embedded in the payload. So transfer exactly `msg_size` bytes and rely on `ncclGin_SignalInc` for the handshake. Document this 8-byte difference in the README.

---

## 2. NCCL GIN device API you will use

All declared in `/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor/nccl/src/include/nccl_device/gin.h` (include via `nccl_device.h`). The user-facing object is `ncclGin` (alias for `ncclGin_BackendMask<ALL>`).

Construct (one context = one QP per peer — matches single-channel):
```cpp
ncclGin gin{devComm, /*contextIndex=*/0};
```

**Put with remote signal increment** (this is the core op). Signature (templated; defaults shown in header):
```cpp
gin.put(
    ncclTeamWorld(devComm), /*peer=*/peer,
    dstWin, dstOffset,      // remote (peer) receive buffer
    srcWin, srcOffset,      // local source buffer
    bytes,
    ncclGin_SignalInc{sig}, // RemoteAction: increment peer's signal `sig` on completion
    ncclGin_None{},         // LocalAction
    coop);                  // ncclCoopThread() or ncclCoopWarp()
```
Key semantics: **`ncclGin_SignalInc` is a *remote* action** — a put from A→B increments **B's** signal. That is exactly what you want for ping-pong handshaking.

**Wait for a signal** (rolling comparison; `least` is the threshold):
```cpp
gin.waitSignal(coop, sig, /*least=*/expected, /*bits=*/64, cuda::memory_order_acquire);
```

**Read / reset signal** (establish a baseline; signals accumulate):
```cpp
uint64_t base = gin.readSignal(sig);     // after reset, 0
gin.resetSignal(sig);                    // host-side setup ensures clean start; call once before loop if needed
```

**Flush** (local completion: source buffers safe to reuse; does *not* guarantee remote settle):
```cpp
gin.flush(coop, cuda::memory_order_acquire);
```

**Initial cross-rank rendezvous barrier** (so both ranks are in the kernel before the first put):
```cpp
ncclGinBarrierSession<ncclCoopCta> bar{ ncclCoopCta(), gin, ncclTeamTagWorld(), /*barrierIdx=*/0 };
bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
// ... ping-pong ...
bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
```

### Coop scope mapping
| NIXL level | Launch | NCCL coop for put/wait |
|------------|--------|------------------------|
| THREAD | `<<<1,1>>>` | `ncclCoopThread()` |
| WARP | `<<<1,32>>>` | `ncclCoopWarp()` |

(Barrier session uses `ncclCoopCta()`; for these tiny launches the CTA == 1 thread or 1 warp.)

---

## 3. Host setup (mirror `02_alltoall_gin`)

Base the host scaffolding on the working example at:
`/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor/nccl/docs/examples/06_device_api/02_alltoall_gin/main.cu`

Reuse the shared harness (gives MPI **and** pthread launch for free):
- `docs/examples/common/include/utils.h` → `int run_example(argc, argv, void*(*)(int my_rank,int total_ranks,int local_device,int devices_per_rank))` and `util_broadcast(...)`.
- `docs/examples/common/include/nccl_utils.h` → `NCCLCHECK`, `CUDACHECK`.

Required host steps (same as the example):
1. `ncclGetUniqueId` on rank 0, `util_broadcast`, `ncclCommInitRank`.
2. `ncclCommQueryProperties` → assert `props.deviceApiSupport` and `props.ginType != NCCL_GIN_TYPE_NONE`. **Also print `props.ginType` and assert it equals the backend forced by `NCCL_GIN_TYPE`** (catches silent GDAKI→proxy fallback).
3. `ncclMemAlloc` send/recv buffers; `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` → `send_win`, `recv_win`.
4. `ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;`
   - `reqs.worldGinBarrierCount = 1;`
   - `reqs.ginSignalCount = 1;`
   - `reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;`
   - `ncclDevCommCreate(comm, &reqs, &devComm);`
5. Allocate a small device buffer for the elapsed-cycles output (`uint64_t* d_elapsed`) and the per-phase sample arrays if you implement phase stats (optional v1; see §5).
6. For each `msg_size` in the sweep: launch the kernel on this rank's stream, `cudaStreamSynchronize`, copy back elapsed cycles, convert to µs, print.
7. Cleanup mirrors the example (`ncclDevCommDestroy`, `ncclCommWindowDeregister`, `ncclMemFree`, finalize/destroy).

**Cycle→µs conversion (host), identical to NIXL:**
```cpp
int clock_khz = 0;
cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, local_device);
double clock_hz = (double)clock_khz * 1000.0;
double rtt_us = (double)elapsed_cycles / (double)iters / clock_hz * 1e6;
double one_way_us = rtt_us / 2.0;
```

---

## 4. Kernel design (the core)

Two ranks (rank 0 = sender, rank 1 = receiver). Both launch the same kernel; branch on `devComm.rank`. Use a single signal index `sig = 0`.

Handshake invariant per iteration `i` (0-based): after `i+1` completed exchanges, each rank's local signal equals `i+1`.

Proposed kernel (new code — adapt freely, keep the protocol):

```cpp
template <typename Coop>
__global__ void gin_pingpong_kernel(
    ncclDevComm devComm,
    ncclWindow_t sendWin, ncclWindow_t recvWin,
    size_t bytes, uint64_t warmup, uint64_t iters,
    uint64_t* d_elapsed /* sender writes total cycles */) {

  Coop coop{};
  const int sig = 0;
  const int peer = devComm.rank ^ 1;          // 2 ranks: 0<->1
  const bool is_sender = (devComm.rank == 0);

  ncclGin gin{devComm, 0};

  // Rendezvous so both ranks are live before the first put.
  ncclGinBarrierSession<ncclCoopCta> bar{ ncclCoopCta(), gin, ncclTeamTagWorld(), 0 };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  const uint64_t total = warmup + iters;
  uint64_t start = 0;

  for (uint64_t i = 0; i < total; ++i) {
    if (is_sender && i == warmup) start = clock64();   // begin timing after warmup

    if (is_sender) {
      // 1) put payload to receiver; increments RECEIVER's signal
      gin.put(ncclTeamWorld(devComm), peer,
              recvWin, 0, sendWin, 0, bytes,
              ncclGin_SignalInc{sig}, ncclGin_None{}, coop);
      // 2) ensure local source is safe / op issued
      gin.flush(coop, cuda::memory_order_acquire);
      // 3) wait for receiver's reply (increments OUR signal to i+1)
      gin.waitSignal(coop, sig, i + 1, 64, cuda::memory_order_acquire);
    } else {
      // receiver: wait for sender's put (our signal reaches i+1)
      gin.waitSignal(coop, sig, i + 1, 64, cuda::memory_order_acquire);
      // reply back to sender; increments SENDER's signal
      gin.put(ncclTeamWorld(devComm), peer,
              recvWin, 0, sendWin, 0, bytes,
              ncclGin_SignalInc{sig}, ncclGin_None{}, coop);
      gin.flush(coop, cuda::memory_order_acquire);
    }
  }

  if (is_sender) {
    uint64_t end = clock64();
    if (/* leader lane */ true) *d_elapsed = end - start;
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}
```

Launch helpers:
```cpp
// THREAD
gin_pingpong_kernel<ncclCoopThread><<<1,1,0,stream>>>(devComm, sendWin, recvWin, bytes, warmup, iters, d_elapsed);
// WARP
gin_pingpong_kernel<ncclCoopWarp><<<1,32,0,stream>>>(devComm, sendWin, recvWin, bytes, warmup, iters, d_elapsed);
```

**Warp-level care:** with `ncclCoopWarp`, all 32 lanes call `put`/`waitSignal`/`flush` cooperatively. Only have **lane 0** read `clock64()` / write `d_elapsed`, and `__syncwarp()` around the timing reads (mirror NIXL).

### Phase breakdown (recommended, matches NIXL `issue`/`complete`/`peer-wait`)
For parity with NIXL's per-phase table, bracket the sender's three steps with `clock64()`:
- `issue` = around step 1 (`put`)
- `complete` = step 2 boundary (`flush` return) − step 1 end
- `peer-wait` = `waitSignal` return − `flush` return
- `rtt` = full loop body
Record per-iteration samples into device arrays (sized `iters`), copy back, compute avg/p50/p90/p99/min/max/stddev on the host. This mirrors NIXL's `record_cycle_sample` + host percentile computation. **v1 may ship RTT-only**; phases are a fast follow.

---

## 5. CLI / parameters (match NIXL & NVSHMEM)

Accept (with these defaults):
| Flag | Default | Meaning |
|------|---------|---------|
| `--msg-size <bytes>` | 8 | single size; **or** `--size-sweep` |
| `--size-sweep` | off | run `8 64 512 4096 32768 262144 1048576` |
| `--iters <n>` | 5000 | timed iterations |
| `--warmup <n>` | 500 | warmup iterations |
| `--warp` | off | WARP level (else THREAD) |
| `--gpu <id>` | from harness/local rank | device id |

(The shared harness passes `local_device`; honor it for multi-process runs.)

---

## 6. File layout to create

**Recommended (reuses harness, lowest friction):**
```
/lustre/fsw/.../nccl/docs/examples/06_device_api/04_gin_pingpong_latency/
├── main.cu        # host setup + kernel + arg parsing + reporting
├── Makefile       # copy 02_alltoall_gin/Makefile, set TARGET = gin_pingpong_latency
└── README.md      # build/run, metric definitions, payload-size note, backend env vars
```
Copy `02_alltoall_gin/Makefile` verbatim and change only `TARGET`. It already wires `common/src/utils.cc`, `NVCC`, MPI/pthread.

> The earlier high-level plan referred to `nccl/contrib/gin_pingpong/`. Prefer the examples location above because it reuses `run_example`/`util_broadcast`/`NCCLCHECK`. If it must live in `contrib/`, you have to replicate the harness (init, broadcast, MPI/pthread) yourself — more work, no benefit.

---

## 7. Build

```bash
cd /lustre/fsw/.../nccl/docs/examples/06_device_api/04_gin_pingpong_latency
# pthread (single node, multi-GPU)
make NCCL_HOME=/path/to/nccl/build CUDA_HOME=/usr/local/cuda
# MPI (multi-node) — required for two-node runs
make MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl/build CUDA_HOME=/usr/local/cuda
```
NCCL must be **built with GIN enabled** (`NCCL_GIN_PROXY_ENABLE=1`, and GDAKI via the gdaki CMake/`makefiles/common.mk` flags). Confirm `nccl_device.h` and a device-API-capable `libnccl` are under `NCCL_HOME`.

---

## 8. Run

Each invocation = **2 ranks**, one GPU per node.

**Validate first on a single node (2 GPUs), proxy backend:**
```bash
NCCL_GIN_TYPE=2 NTHREADS=2 ./gin_pingpong_latency --msg-size 8 --iters 5000 --warmup 500
```

**Two nodes, kernel-initiated (GDAKI):**
```bash
NCCL_GIN_TYPE=3 mpirun -n 2 --host node0,node1 \
  -x NCCL_GIN_TYPE -x LD_LIBRARY_PATH \
  ./gin_pingpong_latency --size-sweep --iters 5000 --warmup 500
```

**Two nodes, CPU-proxy (GIN proxy):**
```bash
NCCL_GIN_TYPE=2 mpirun -n 2 --host node0,node1 \
  -x NCCL_GIN_TYPE -x LD_LIBRARY_PATH \
  ./gin_pingpong_latency --size-sweep --iters 5000 --warmup 500
```
Add `--warp` for warp-level. Force a single NIC/HCA and pin NUMA so all stacks use the same path.

---

## 9. Output format (must be parseable & comparable)

Print on the **sender** (rank 0) one summary line per (size, level), matching NIXL's shape so a shared parser works:
```
op=put  backend=gin_proxy  level=THREAD  msg_size=8  iters=5000  RTT=12.300 us  one-way=6.150 us
```
Where `backend` ∈ {`gin_proxy`,`gdaki`} (derive from `props.ginType`), `level` ∈ {`THREAD`,`WARP`}.

If phase stats are implemented, also emit (mirroring NIXL `[pingpong-stats]`):
```
[pingpong-stats] issue     n=5000 avg=.. p50=.. p90=.. p99=.. min=.. max=.. std=..
[pingpong-stats] complete  n=5000 avg=..
[pingpong-stats] peer-wait n=5000 avg=..
[pingpong-stats] rtt       n=5000 avg=..
```
Also append a CSV row to a file given by `--csv <path>` with columns matching the cross-stack schema:
```
stack,backend,class,level,msg_size,iters,one_way_us,rtt_us
nccl,gin_proxy,proxy,THREAD,8,5000,6.150,12.300
nccl,gdaki,kernel,THREAD,8,5000,4.900,9.800
```
(`class` = `proxy` for gin_proxy, `kernel` for gdaki.)

---

## 10. Correctness validation (do this before trusting latencies)

- Initialize `sendWin` with a known pattern; after the run, verify the receiver's `recvWin` holds the sender's last payload (and vice versa). A latency benchmark that silently drops puts will report bogus-fast numbers.
- Assert `waitSignal` actually advanced (e.g. final `readSignal(sig) == iters + warmup`).
- Run a tiny `--iters 5 --warmup 1` correctness pass with `NCCL_DEBUG=INFO` and confirm the expected `ginType` and no fallback warnings.

---

## 11. Pitfalls & gotchas (read before coding)

1. **Signal direction.** `ncclGin_SignalInc{sig}` increments the **peer's** signal, not the local one. The sender waits on its **own** signal being bumped by the receiver's reply. Getting this backwards deadlocks.
2. **Rolling comparison.** `waitSignal(..., least, bits, ...)` uses rolling (wrap-safe) comparison. Use `bits=64` and absolute thresholds `i+1`; do not let the threshold lap the 64-bit space (won't happen at realistic iters).
3. **Clean baseline.** Ensure signals start at 0 for the timed region. The device comm is fresh per run, but if you reuse it across sizes, `resetSignal(sig)` on both ranks (and re-barrier) between sizes, or carry a running baseline.
4. **Both kernels must be co-resident.** The receiver must be spinning in `waitSignal` while the sender puts. Launch on each rank's own stream and `cudaStreamSynchronize` after; do not serialize the two ranks.
5. **`flush` ≠ remote completion.** `flush` only guarantees local source reuse safety. Remote arrival is observed via the signal. Keep `complete` (flush) and `peer-wait` (signal) as separate phases — do not conflate.
6. **Warp level.** Only lane 0 times and writes output; `__syncwarp()` around timing. All lanes must call the collective `put`/`wait`/`flush`.
7. **Backend verification.** `NCCL_GIN_TYPE=3` (GDAKI) **silently falls back to proxy** if MLX5/DOCA prerequisites aren't met. Assert `props.ginType` matches the request and fail loudly otherwise — otherwise you'll mislabel a proxy run as GDAKI.
8. **Single QP parity.** Use exactly one GIN context (index 0) = one QP per peer, to match NIXL's single channel and a single-QP NVSHMEM config. Don't add extra contexts.
9. **Payload vs NIXL +8 bytes.** Transfer exactly `msg_size` (signal carries the handshake). Note this in the README so the comparison's small-size points are interpreted correctly.
10. **NUMA/NIC pinning.** Pin the same HCA and CPU NUMA node across all three stacks; an unpinned proxy thread can add microseconds and unfairly penalize the proxy row.

---

## 12. Acceptance criteria (definition of done)

- [ ] `main.cu`, `Makefile`, `README.md` created under `06_device_api/04_gin_pingpong_latency/`.
- [ ] Builds in both pthread and `MPI=1` modes without warnings.
- [ ] Runs 2 ranks; correctness check passes (payload verified, signals advanced).
- [ ] Works with `NCCL_GIN_TYPE=2` and `NCCL_GIN_TYPE=3`; asserts the engaged backend.
- [ ] THREAD and WARP levels both supported (`--warp`).
- [ ] Size sweep `8 … 1 MiB`; defaults `--iters 5000 --warmup 500`.
- [ ] Emits the parseable summary line + CSV row (schema in §9); RTT and one-way in µs via `clock64`+`cudaDevAttrClockRate`.
- [ ] (Recommended) per-phase `issue`/`complete`/`peer-wait`/`rtt` stats.
- [ ] README documents build, run (single + two node), metric definitions, the +8-byte payload note, and backend env vars.

---

## 13. Reference files

NCCL (in `/lustre/fsw/.../nccl`):
- `docs/examples/06_device_api/02_alltoall_gin/main.cu` — host setup + `gin.put`+signal+flush+barrier (closest template).
- `docs/examples/06_device_api/02_alltoall_gin/Makefile` — copy for the new target.
- `docs/examples/06_device_api/02_alltoall_gin/README.md` — barrier/signal/window explanation.
- `docs/examples/common/include/utils.h` / `src/utils.cc` — `run_example`, `util_broadcast`.
- `docs/examples/common/include/nccl_utils.h` — `NCCLCHECK`, `CUDACHECK`.
- `src/include/nccl_device/gin.h` — full `ncclGin` device API (put/waitSignal/readSignal/flush/barrier).
- `src/transport/net_ib/gin.cc` — `NCCL_GIN_TYPE` param + GDAKI-first/proxy-fallback selection.

NIXL reference protocol (in `nixl-cpu-proxy`, for fidelity only):
- `examples/device/pingpong/bench_kernel.cu` — `clock64` timing, phase recording, ping-pong body.
- `examples/device/pingpong/bench_main.cpp` — arg parsing, cycle→µs, `[pingpong-stats]` printing.

Cross-stack methodology (the parent plan this task feeds into):
- `nixl-cpu-proxy/docs/PUT_LATENCY_COMPARISON.md` — backend mapping, conditions, aggregation schema (Step E), caveats.
```
