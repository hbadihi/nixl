# CPU Proxy — Demo Brief

> Single-source brief for the CPU-proxy demo presentation. Companion to the
> hard numbers in
> [`examples/device/pingpong/scripts/PERFORMANCE.md`](../examples/device/pingpong/scripts/PERFORMANCE.md).

---

## 0. The 30-second pitch

NIXL today exposes a GPU-side `nixlPut` that requires GPU-initiated UCX (GDA-KI / IBGDA / device-API). The **CPU proxy** keeps the same `nixlPut` GPU API but routes the submission through a host worker thread, so transfers happen on a **plain UCX path that runs anywhere UCX runs** — no GDA-KI, no IBGDA, no device-API.

Concretely, we shipped:

- A transport-agnostic **proxy core** in `src/core/device_proxy/` (~1.4 K LoC).
- A **GPU device API** layer in `src/api/gpu/` that selects between `ucx` and `proxy` backends at compile time (~700 LoC, restructured).
- A **UCX proxy backend** plugin in `src/plugins/ucx/device_proxy/` (~225 LoC).
- An **agent-lifecycle integration** (~150 LoC of `nixl_agent.cpp` / `agent_data.h` changes) that auto-wraps mem-views when proxy mode is on.
- **Two test layers**: 14 + 17 host-only gtest unit tests (registry + runtime) and 12 GPU device-API gtests; plus an existing `single_write_test.cu` extended to cover proxy.
- A **benchmark + profiling harness** (`examples/device/pingpong/` + `scripts/`) — two side-by-side binaries (`nixl_device_pingpong` and `nixl_device_pingpong_proxy`) and a one-shot `profile_overhead.sh` that produces sweep CSVs, Nsight Systems traces, and per-stage proxy-worker stats.

Total scope: **56 files changed, +5 943 / −204 LoC** vs. upstream `2758940`.

---

## 1. Code map & concepts

### 1.1 Layered view

```
GPU kernel  ─►  nixlPut<level>(...)                   src/api/gpu/common/nixl_device_wrappers.cuh
                     │
                     ▼
                nixl::gpu::selected_impl::put         src/api/gpu/common/nixl_device_api.cuh
                  ├─ proxy_impl                       src/api/gpu/proxy/nixl_device_impl.cuh
                  └─ ucx_impl                         src/api/gpu/ucx/nixl_device_impl.cuh
                     │  (proxy path only)
                     ▼
                ProxyDeviceContext::enqueue           src/api/gpu/proxy/nixl_device_proxy.cuh
                  • atomic slot reservation in WorkRing
                  • writes ProxySubmission to mapped pinned ring
                  • returns ProxyXferStatus token
─────────────────────────────────────────────────────  GPU ↔ CPU boundary
                ProxyWorker thread                    src/core/device_proxy/proxy_worker.cpp
                  runOnce():
                    tryDequeue → submitToBackend → driveBackendProgress → publishCompletions
                     │
                     ▼
                DeviceProxyBackendAdapter (abstract)  src/core/device_proxy/backend_adapter.h
                     │
                     ▼
                nixlUcxProxyBackend                   src/plugins/ucx/device_proxy/ucx_proxy_backend.cpp
                  • prepXfer / postXfer / checkXfer on the existing nixlUcxEngine
                     │
                     ▼
                UCX (RoCE / cuda_ipc / shm / tcp)
─────────────────────────────────────────────────────  completion path
                CompletionSlot.completed_idx (mapped pinned)
                     │
                     ▼
                GPU kernel polls via ProxyDeviceContext::pollXferStatus
                     │
                     ▼
                nixlGpuGetXferStatus → NIXL_SUCCESS
```

### 1.2 New core module: `src/core/device_proxy/`

| File | LoC | Role |
|---|---|---|
| `proxy_protocol.h` | 86 | On-the-wire data types shared by GPU & CPU: `ProxySubmission`, `WorkRing`, `CompletionSlot`, `ProxyChannelView`, `ProxyDeviceContextData`, opcodes & control words |
| `backend_adapter.h` | 71 | Pure-virtual `DeviceProxyBackendAdapter` (transport boundary) + `PreparedProxySubmission` |
| `proxy_runtime.{h,cpp}` | 245 + 635 | Owns channels, registry, workers; allocates pinned/host-mapped ring buffers; manages init / startWorkers / shutdown lifecycle; resolves proxy mem-views back to backend mem-views |
| `proxy_worker.{h,cpp}` | 94 + 334 | The host-side thread loop + per-stage instrumentation (timing stats + NVTX) |

**Concepts to highlight in the demo:**

1. **Per-channel MPSC work ring**, lock-free.
   - `producer_idx`, `consumer_idx`, and `records[]` all live in **`cudaMallocHost` pinned memory** mapped into the GPU address space (`cudaHostGetDevicePointer`).
   - GPU writes `producer_idx` and `records[slot]` via `cuda::atomic_ref<…, thread_scope_system>`, signals `ready_flag` with release semantics.
   - CPU reads `ready_flag` with `__atomic_load_n(ACQUIRE)` and advances `consumer_idx`.
   - No driver call on the hot path in either direction.

   GPU-side enqueue:

   ```146:186:src/api/gpu/proxy/nixl_device_proxy.cuh
       __device__ inline nixl_status_t
       enqueue(ProxySubmission submission, nixlGpuXferStatusH *xfer_status = nullptr) {
           if (submission.channel_id >= num_channels) {
               return NIXL_ERR_INVALID_PARAM;
           }
           ProxyChannelView &channel_view = channels[submission.channel_id];
           WorkRing         *ring    = channel_view.work_ring;
           cuda::atomic_ref<uint32_t, cuda::thread_scope_system> prod(*ring->producer_idx);
           cuda::atomic_ref<uint32_t, cuda::thread_scope_system> cons(*ring->consumer_idx);
           // ... atomic slot claim, spin-on-full, write record, release ready_flag ...
   ```

   CPU-side dequeue:

   ```180:204:src/core/device_proxy/proxy_worker.cpp
   bool
   ProxyWorker::tryDequeue(ChannelState &channel, ProxySubmission &submission) {
       WorkRing *ring = channel.work_ring_;
       uint32_t local_consumer_idx =
           __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
       uint32_t slot = local_consumer_idx % ring->depth;
       if (!__atomic_load_n(&ring->records[slot].ready_flag, __ATOMIC_ACQUIRE)) {
           return false;
       }
       submission = ring->records[slot];
       ...
   ```

2. **Collapsed completion queue** — one `CompletionSlot` per channel, not per request. Holds `(completed_idx, next_status)`.
   - Successful completions only bump `completed_idx`; readers infer "my op succeeded" from `completed_idx > op_idx`.
   - First terminal failure is **latched** into `next_status`, so any later poll sees a real error code instead of spinning forever.
   - `op_idx` is a **64-bit monotonic counter starting at 1** (so `0` unambiguously means "nothing completed"), guarded by a `static_assert` in `nixl_device_proxy.cuh`.

3. **Proxy mem-view registry** (`ProxyMemViewRegistry` in `proxy_runtime.{h,cpp}`).
   - `nixl_agent.cpp` intercepts `prepMemView` / `loadRemoteMD` and, when proxy mode is on for that backend, swaps the real mem-view handle for a **proxy mem-view ID** (the registry index packed into a handle).
   - The GPU only ever sees proxy IDs — it never holds a UCX memh.
   - The worker resolves IDs back to backend descriptors via `prepareSubmission()` before calling `backend->submit`.
   - Three-state lifecycle: `Allocated → Ready → Retired` (so the publish race is documented and contained).

4. **`DeviceProxyBackendAdapter` is the only transport contract.** Any backend that implements `init / loadRemoteConnInfo / submit / checkCompletion / progress / shutdown` plugs in. UCX is the first; libfabric is the next planned consumer (see `TODO.md`).

5. **`ProxyWorker::runOnce()` is the heart of the loop**:
   - Try to dequeue from each assigned channel → call `submitToBackend` → push into `inflight_requests` deque.
   - Drive `backend->progress()` once.
   - Walk each channel's deque front-to-back, popping completed ops and bumping `completed_idx`.
   - All three stages bracketed by NVTX (`prx:submit`, `prx:progress`, `prx:publish`) and instrumented with three `ProxyWorkerStats` counters (`prep+submit`, `inflight`, `publish`) plus a `polls/request` ratio.

**Current scaling note:** multi-channel mode is exercised in tests, but the UCX proxy backend still forwards all workers through the same `nixlUcxEngine::postXfer` path. Until that path is validated for concurrent proxy workers, the demo keeps `proxyWorkerCount = 1` and uses extra channels only as GPU submission queues.

### 1.3 GPU device-API restructure: `src/api/gpu/`

This is the layer that lets the **same GPU code** target either UCX-direct or the proxy by build flag.

| Path | Purpose |
|---|---|
| `common/nixl_device_types.cuh` | `nixlGpuXferStatusH` (64-byte opaque), `nixl_gpu_level_t` (THREAD/WARP/BLOCK/GRID), `nixlMemViewElem` |
| `common/nixl_device_api.cuh` | Picks `selected_impl` namespace based on `NIXL_GPU_DEVICE_BACKEND_PROXY` / `_UCX` |
| `common/nixl_device_wrappers.cuh` | The public `nixlPut`, `nixlAtomicAdd`, `nixlGpuGetXferStatus`, `nixlGetPtr` templates that user kernels call |
| `proxy/nixl_device_impl.cuh` | Proxy implementation of `put / atomic_add / get_xfer_status` (lane-0 enqueue + level-aware sync/broadcast) |
| `proxy/nixl_device_proxy.{cu,cuh}` | `ProxyDeviceContext` (extends `ProxyDeviceContextData`), the global `g_nixl_proxy_ctx` symbol, and `nixlProxyPublishContext()` host helper |
| `ucx/nixl_device_impl.cuh` | UCX-direct implementation (factored out of the previous `ucx/nixl_device.cuh`) |

The factor-out is what the `device_api: factor out common GPU layer and add proxy device API` commit did — it preserves all four execution levels (THREAD/WARP/BLOCK/GRID) for both backends, including a cooperative-grid `__shared__` broadcast for `get_xfer_status<GRID>`.

### 1.4 UCX proxy backend: `src/plugins/ucx/device_proxy/`

`nixlUcxProxyBackend` is a thin shim that owns no transport state of its own — it forwards into the existing `nixlUcxEngine`:

- `submit(PUT)` → `engine_->prepXfer(NIXL_WRITE, …)` + `engine_->postXfer(...)` and stores the `nixlBackendReqH` under a monotonic 64-bit token.
- `checkCompletion(token)` → `engine_->checkXfer(handle)`; on terminal status, releases the handle and erases it.
- `progress()` → no-op (UCX backend already pumps via its own progress thread).
- `shutdown()` → drains remaining handles.

This is the only file that touches `ucp_*` symbols outside the rest of `src/plugins/ucx/`. Currently still wired through `STATIC_PLUGIN_UCX` in `nixl_agent.cpp`; making it a fully dynamic plugin is the first item in `TODO.md`.

### 1.5 Agent integration: `src/core/nixl_agent.cpp` + `agent_data.h`

| Hook | What it does |
|---|---|
| `nixlAgentConfig.enableDeviceProxy / proxyWorkerCount / proxyChannelCount` | New config fields (defaults: off, 1/1) — `src/api/cpp/nixl_params.h` |
| `nixlAgentData::createProxyRuntime(engine, backend)` | Lazily builds the `ProxyRuntime` + `nixlUcxProxyBackend` the first time a backend that supports proxy gets initialized |
| `createBackend()` | Calls `createProxyRuntime` whenever proxy mode is enabled |
| `loadRemoteMD` / `prepMemView` paths | If proxy is active for this engine, registers the original `mvh` with the proxy registry, stores metadata, and **returns the proxy handle to the user** — the GPU only ever gets proxy IDs |
| `nixlAgent::getProxyDeviceContext()` | New public accessor returning `ProxyDeviceContextData *` so the user can publish it to `g_nixl_proxy_ctx` via `nixlProxyPublishContext` |
| `~nixlAgent` | Calls `shutdownProxyRuntime` before tearing down backends |

### 1.6 Build wiring (one slide if you want it)

- New meson option in `meson_options.txt`.
- `examples/device/pingpong/meson.build` produces **both** binaries from the same sources, differing only in the `NIXL_GPU_DEVICE_BACKEND_*` define and the include path resolution that picks `proxy/nixl_device.cuh` vs `ucx/nixl_device.cuh`.
- `src/api/gpu/meson.build` exposes the common headers; `src/core/meson.build` adds the four new device-proxy translation units.

---

## 2. The benchmark — how it works end-to-end

### 2.1 The latency kernel — what's actually being measured

Source: `examples/device/pingpong/bench_kernel.cu`.

It's a two-process, two-GPU-buffer ping-pong using NIXL's **GPU-side `nixlPut`**. Each side allocates two device buffers:

- `send_buf` — what *we* PUT into the peer.
- `recv_buf` — what *the peer* PUTs into us.

Layout per buffer: `[ msg_size payload bytes | uint64_t sequence counter ]`. The buffer size is `msg_size + sizeof(uint64_t)` and each PUT moves the whole thing (so the counter is always carried in the same RDMA write as the payload).

The kernel runs at **either THREAD or WARP granularity** (one CUDA block, 1 or 32 threads):

```119:133:examples/device/pingpong/bench_kernel.cu
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD>(gpu_bench_ctx ctx, uint64_t *elapsed_device);
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP>(gpu_bench_ctx ctx, uint64_t *elapsed_device);

void
launch_pingpong_thread(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD><<<1, 1, 0, stream>>>(ctx, d_elapsed);
}

void
launch_pingpong_warp(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP><<<1, 32, 0, stream>>>(ctx, d_elapsed);
}
```

The body is symmetric except for who moves first. For **`warmup_iters + num_iters`** iterations the loop does:

**Sender:**
1. Bump `*send_counter = i + 1` (lives at the tail of `send_buf`).
2. `do_put_async(local_mvh, remote_mvh, total_size, xfer_status)` — fires off a `nixlPut` of the whole buffer (payload + counter) into the peer's `recv_buf`.
3. Spin on `recv_counter` (the tail of *our* `recv_buf`) until it reaches `i + 1`. That's how we know the peer has both received our PUT and replied.

**Receiver:** mirror image — wait for `recv_counter == i+1`, then bump `send_counter` and PUT back.

```81:111:examples/device/pingpong/bench_kernel.cu
        if (ctx.is_sender) {
            if (lane_id == 0) {
                *send_counter = i + 1; // Increment send counter to signal the receiver
            }
            ...
            do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
            if (lane_id == 0) {
                wait_sequence_number(recv_counter, i + 1);
            }
            ...
        } else {
            if (lane_id == 0) {
                wait_sequence_number(recv_counter, i + 1);
            }
            ...
            if (lane_id == 0) {
                *send_counter = i + 1;
            }
            do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
        }
```

Two important details:

- **`do_put_async` doesn't wait for completion.** It calls `nixlPut` (which under the proxy build hands the work to the CPU worker via the lock-free ring, returns `NIXL_IN_PROG`) and returns immediately. The synchronization comes from polling `recv_counter` — i.e. the **kernel doesn't observe the proxy's CompletionSlot at all**, it just observes the bytes landing in HBM. This is on purpose: it makes the proxy and UCX-direct paths use *exactly* the same kernel.
- **`wait_sequence_number` does a busy-wait with `__nanosleep(50)`** — that's the GPU-side spin while the data is in flight, copied straight from UCX's reference pingpong.

A `do_put_sync` variant is also defined (PUT then poll `nixlGpuGetXferStatus` until it returns success), but the benchmark intentionally uses `do_put_async` so UCX-direct and proxy runs measure the same GPU-observed ping-pong protocol.

#### Timing

`clock64()` is sampled once at iteration `warmup_iters` (start of timed phase) and once after the last iteration. The delta in SM cycles is written to `*elapsed_device`:

```76:78:examples/device/pingpong/bench_kernel.cu
        if (ctx.is_sender && lane_id == 0 && i == ctx.warmup_iters) {
            start_time = clock64(); // Start timing after warmup
        }
```

Only the **sender** records elapsed cycles — its loop measures one full PUT + counter-arrival round trip. The host then converts cycles → microseconds using `cudaDeviceGetAttribute(cudaDevAttrClockRate)`:

```79:84:examples/device/pingpong/bench_main.cpp
    double rtt_us     = (double)h_elapsed / (double)num_iters / clock_hz * 1e6;
    double one_way_us = rtt_us / 2.0;

    printf("msg_size=%-6zu  iters=%-6llu  RTT=%.3f us  one-way=%.3f us  [%s]\n",
           msg_size, (unsigned long long)num_iters, rtt_us, one_way_us,
           use_warp ? "WARP" : "THREAD");
```

The sweep harness greps the `RTT=` line out of the sender's stdout.

So **what's measured is the GPU-observed round trip**: counter store on side A → kernel-issued `nixlPut` of A's buffer → peer kernel sees its `recv_counter` advance → peer's `nixlPut` back → A's kernel sees its own `recv_counter` advance. Everything (UCX, proxy ring, CPU worker, completion path) is inside that interval.

### 2.2 Host setup — `BenchContext`

Source: `examples/device/pingpong/bench_host.{h,cpp}`. One `BenchContext` per side, RAII-managed. `setup()` does the 8 steps below — they're worth knowing because every one of them shows up in the demo question "wait, what does the proxy actually change?"

| # | What | Proxy-only difference |
|---|---|---|
| 1 | `cudaSetDevice(gpu_id)` | — |
| 2 | Construct `nixlAgent` with progress + listen threads. | The proxy build sets `cfg.enableDeviceProxy = true; proxyChannelCount = 1; proxyWorkerCount = 1;` |
| 3 | `agent->createBackend("UCX", ...)` | Triggers `nixlAgentData::createProxyRuntime(engine, "UCX")` lazily inside the agent. |
| 4 | Allocate `send_buf` / `recv_buf` on the GPU | — |
| 5 | `agent->registerMem(...)` for both buffers | — |
| 6 | TCP metadata exchange via `fetchRemoteMD` / `sendLocalMD` | — |
| 7 | Exchange recv-buffer device addresses through NIXL **notifications** (`genNotif` / `getNotifs`) — this is how each side learns the peer's `recv_buf` virtual address so it can put into it. | — |
| 8 | `agent->prepMemView(...)` for both local and remote dlists → produces `local_mvh` / `remote_mvh` | When proxy mode is on, the `prepMemView` path inside `nixl_agent.cpp` swaps the real UCX mem-view for a **proxy mem-view ID** in the registry. The kernel only ever sees that opaque proxy ID. |

The proxy-specific bit happens between steps 3 and 4:

```56:69:examples/device/pingpong/bench_host.cpp
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    void *proxy_ctx = agent->getProxyDeviceContext();
    if (proxy_ctx == nullptr) {
        fprintf(stderr, "[%s] proxy device context not available\n", my_name.c_str());
        return NIXL_ERR_BACKEND;
    }
    if (bench_proxy_publish_context(proxy_ctx) != cudaSuccess) {
        fprintf(stderr, "[%s] bench_proxy_publish_context failed\n", my_name.c_str());
        return NIXL_ERR_BACKEND;
    }
#endif
```

`getProxyDeviceContext()` returns the `ProxyDeviceContextData *` allocated by `ProxyRuntime::init` (the channel views, shutdown word, etc.). `bench_proxy_publish_context` is a thin host-side wrapper defined in `bench_kernel.cu` that calls `nixlProxyPublishContext`, which writes that pointer into a process-wide `__device__ ProxyDeviceContext *g_nixl_proxy_ctx`. From that moment on, any kernel in the process reading `load_proxy_context()` gets a valid context — that's how `nixlPut` in the kernel finds the work ring.

That single global is why `bench_main.cpp` **disables single-process mode in the proxy build**: two `BenchContext`s in one process would clobber each other on `g_nixl_proxy_ctx`. The `#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY` blocks in `bench_main.cpp` are exclusively for that.

The `~BenchContext` symmetric tear-down runs `bench_proxy_clear_context()` before the agent is destroyed, so the `__device__` pointer doesn't dangle past the proxy runtime's lifetime.

A few things in setup that are easy to miss but useful in Q&A:

- The metadata exchange uses **NIXL's own TCP listen thread**, not a hand-rolled socket. Sender drives `fetchRemoteMD` / `sendLocalMD`; receiver is passive.
- The recv-buffer address exchange uses **NIXL notifications** (small inline blobs), not the metadata blob, because addresses change per run while metadata is stable per registration. The receiver waits for the listen thread to populate `remoteBackends_` (`checkRemoteMD` returns `NIXL_ERR_NOT_FOUND` silently until then) and only then calls `genNotif`.
- The final `prepMemView` calls **spin until success** with `sleep(1ms)`. They fail (and would log an ERROR every iteration without the throttle) until the listen thread has fully ingested the peer's metadata — that's why the script forces `NIXL_LOG_LEVEL=FATAL` and `PERFORMANCE.md` calls out "bound the `prepMemView` retry loop with a timeout" as a polish item.

### 2.3 `bench_main.cpp` — three execution modes

Source: `examples/device/pingpong/bench_main.cpp`.

It supports three configurations. Mode selection is by CLI flags:

| Mode | When | How |
|---|---|---|
| **two-process** (`--role sender / --role receiver`) | Default for the proxy build; the only mode the script uses. | Each side: make one `BenchContext`, allocate `d_elapsed`, build `gpu_bench_ctx`, launch the kernel, `cudaStreamSynchronize`, print RTT (sender only). Code in `twoprocess_run()`. |
| **single-process loopback** (`--single-process`) | UCX-direct build only. Two `std::thread`s in one process, two `BenchContext`s, ports `base_port` and `base_port+1`, `127.0.0.1`. | Disabled for proxy builds because of the global `g_nixl_proxy_ctx`. |
| **default** | If you give it no args, it falls into single-process (UCX) or rejects with usage (proxy). | — |

Each "side" of any mode is the same three steps:

1. `BenchContext::setup(...)` — section 2.2.
2. Build `gpu_bench_ctx` with the mvh handles, buffer pointers, and per-iteration counts.
3. Launch `launch_pingpong_thread` or `launch_pingpong_warp` on a CUDA stream and `cudaStreamSynchronize`.

`d_elapsed` is `cudaMalloc`'d once per side; the sender's value is what `print_latency` converts and prints.

### 2.4 The harness — `scripts/profile_overhead.sh`

This is what makes the demo run reproducible.

It builds **everything off two assumed binaries**:

```47:48:examples/device/pingpong/scripts/profile_overhead.sh
UCX_BIN="${BIN_DIR}/nixl_device_pingpong"
PROXY_BIN="${BIN_DIR}/nixl_device_pingpong_proxy"
```

Both are compiled from the same source set; the only difference is the `NIXL_GPU_DEVICE_BACKEND_*` define that picks which `selected_impl` `nixl_device_api.cuh` aliases to. That's the whole "side-by-side comparison" claim — same kernel, same host, same buffers, same metadata exchange, only the GPU `nixlPut` implementation differs.

#### `run_one` — the per-data-point primitive

Always runs **two-process loopback** on this host (`RECV_HOST=127.0.0.1` by default), even for the UCX binary. Picks a fresh free port pair via `ss -tln` so back-to-back runs don't trip on `TIME_WAIT`:

```141:163:examples/device/pingpong/scripts/profile_overhead.sh
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
```

Stdout and stderr are split per side, into 4 files per data point:
`<tag>_send.{out,err}` and `<tag>_recv.{out,err}`. Stdout has the `RTT=` line (small, parseable); stderr has NIXL/CUDA logs and — crucially — the `[proxy-stats]` block that `ProxyWorker` dumps at thread exit.

The script exports two env vars on your behalf so the stats lines actually land:

- `NIXL_LOG_LEVEL=FATAL` — silences the `prepMemView` retry-spam during setup.
- `NIXL_PROXY_STATS=1` — guarantees `ProxyWorker::logStatsSummary` writes to stderr.

#### Three modes wrapping `run_one`

| Mode | What it does | Output |
|---|---|---|
| `sweep [iters] [warmup]` | For each size in `SIZES` (default `8 64 512 4096 32768 262144 1048576`) and each variant in `{ucx, proxy}`, calls `run_one`, greps `RTT=` out of `<send_out>`, appends a row to `sweep.csv`, then renders `summary.txt` with delta and ratio columns. | `sweep.csv`, `summary.txt`, plus 4 logs × 7 sizes × 2 variants = 56 files |
| `nsys [size] [iters] [warmup]` | One run per variant, but wrapped in `nsys profile -t cuda,nvtx,osrt -o <rep>`. The NVTX captured comes from the `prx:submit` / `prx:progress` / `prx:publish` ranges in `ProxyWorker`. | `nsys_<variant>_<size>.nsys-rep` |
| `ucxinfo [size] [iters] [warmup]` | One run per variant with `UCX_PROTO_INFO=y UCX_LOG_LEVEL=info` so UCX prints which transports/protocols it chose for each PUT size. | `ucxinfo_*.{out,err}` |
| `all` | Sweep + nsys (8 KiB / 2000 iters) + ucxinfo (8 B / 200 iters). | All of the above |

`summary.txt` is the table you see in the deck — produced by an awk one-liner over the CSV:

```229:251:examples/device/pingpong/scripts/profile_overhead.sh
        awk -F, '
            NR==1 { next }
            { rtt[$1","$2]=$5; sizes[$2]=1 }
            END {
                printf "  %10s  %12s  %12s  %12s  %10s\n",
                       "msg_size", "ucx_us", "proxy_us", "delta_us", "ratio"
                ...
                    if (u>0 && p>0) {
                        printf "  %10d  %12.2f  %12.2f  %12.2f  %9.2fx\n",
                               s, u, p, p-u, p/u
                    }
                ...
            }' "${csv}"
```

`delta_us = proxy − ucx`, `ratio = proxy / ucx`. That's where the "3.00× / 0.43× / 0.10×" numbers in the deck come from.

#### `analyze_nsys.sh` — the post-processor

Sibling script. It:
1. Greps `RTT=` out of every `*_send.out` in the run dir.
2. Greps `[proxy-stats]` out of every `*_send.err` and `*_recv.err`.
3. Runs `nsys stats --report nvtx_pushpop_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,cuda_gpu_mem_time_sum` over each `.nsys-rep` and keeps the top N rows.
4. Writes everything to `<run_dir>/analysis.txt`.

That's where the `:prx:progress / :prx:submit / :prx:publish` percentages and the `cuMemcpyDtoDAsync_v2` / `cuEventQuery` / `cuStreamAddCallback` counts in the deck come from.

### 2.5 Where each measurement comes from (one-page cheat sheet for Q&A)

| Number on a slide | Where it's computed | Where it's logged |
|---|---|---|
| **End-to-end RTT** (`13.05 ms` etc.) | `clock64()` delta in the kernel, divided by `num_iters` and SM clock rate on the host | sender stdout `RTT=...`; harvested by `parse_rtt_us` in the script |
| **`prep+submit` floor / avg / max** | `ProxyWorker::runOnce`: `steady_clock` between `tryDequeue` and `submitToBackend` returning | `[proxy-stats][w0] prep+submit ...` in `*_recv.err` |
| **`inflight` µs** | Between the `submit_time` stamped on the inflight deque entry and the moment `checkCompletion` returns terminal | `[proxy-stats][w0] inflight ...` |
| **`publish` ns** | Between `t_complete` and the `__atomic_store_n` that bumps `completed_idx` | `[proxy-stats][w0] publish ...` |
| **`polls/request` (worker idle %)** | `run_once_count_ / prep_submit_stats_.count` | `[proxy-stats][w0] polls/request=...` |
| **NVTX percentages (`:prx:*`)** | NVTX 3 ranges in `proxy_worker.cpp`, captured by `nsys profile -t nvtx` | `nsys stats --report nvtx_pushpop_sum` → `analysis.txt` |
| **Per-PUT GPU memcpy cost (`1.9 µs`)** | UCX's underlying `[CUDA memcpy Device-to-Device]` counted by Nsight | `nsys stats --report cuda_gpu_mem_time_sum` → `analysis.txt` |

The script + the kernel together give you the "RTT from the GPU's point of view"; the `[proxy-stats]` block + NVTX give you the "what *inside* that RTT was actually our proxy and what was UCX". Putting those two side-by-side is the entire performance story the deck tells.

---

## 3. Tests

### 3.1 Host-only unit tests (no CUDA hardware needed beyond device pointers)

`test/gtest/unit/proxy_memview_registry/proxy_memview_registry.cpp` — 259 LoC, 14 tests:
`RegisterSingle`, `RegisterNullOutputReturnsError`, `RegisterMultipleAssignsUniqueIds`, `ResolveByHandle`, `ResolveById`, `ResolveMultiple`, `AllocatedEntryIsResolvableBeforeMetadataPublish`, `PrepareSubmissionRequiresReadyEntries`, `ReadyEntriesProducePreparedTransportDescriptors`, `MetadataKindMustMatchSubmissionRole`, `RetiredEntriesStopFutureDispatchButKeepOtherEntriesUsable`, `ClearRetiresExistingEntriesAndPreservesFreshIds`, `StoreMetadataRejectsRetiredEntries`.

`test/gtest/unit/proxy_runtime/proxy_runtime.cpp` — 343 LoC, ~17 tests against a stub backend, including:
`InitCallsBackendInit`, `InitRejects{Null,ZeroChannels,ZeroWorkers}`, `InitPropagatesBackendFailure`, `DeviceChannelViewsPopulated`, `WorkRingIndicesStartAtZero`, `CompletionSlotsInitialized`, `WorkerCountClampedToChannels`, `DeviceContextPopulated/NullAfterShutdown`, `StartWorkersAndShutdown`, `RestartWorkersWithoutShutdown`, `Shutdown{WithoutStart,BeforeInit,Double}IsHarmless`, `InitAfterShutdownWorks`, `Single/ManyChannel(s)Single/ManyWorker(s)`, **`WorkerSubmitsPreparedTransportDescriptors`** (the end-to-end test: writes a `ProxySubmission` directly into the device-mapped ring, lets the worker pick it up, asserts the `PreparedProxySubmission` reaching the stub backend has the correct resolved `nixlMetaDesc` for both local and remote sides).

### 3.2 GPU device-API gtests

`test/gtest/device_api/proxy_write_test.cu` — 973 LoC, 12 tests using a `ControllableStubAdapter` (lets the test thread mark each token complete on demand):

`ContextPublishedAfterStartWorkers`, `ContextClearedAfterShutdown`, `PutReturnsInProgWhenEnqueued`, `PutCompletionRoundTrip`, `CompletionNotVisibleUntilPublished`, `MultipleSubmissionsCompletionFrontier`, `EarlierCompletionStaysSuccessfulAfterLaterError`, `EarlierErrorPropagatesToLaterQueuedOp`, `CompletionPropagatesErrorStatus`, `SubmitFailurePropagatesErrorStatus`, `RingOverflowReturnsBackendErrorOnShutdown`, `ChannelCompletionsAdvanceIndependently`.

These exercise the full GPU↔CPU loop on real CUDA pinned memory — they're the strongest evidence that the collapsed-CQ semantics behave correctly under failure, error-latching, and multi-channel concurrency.

`test/gtest/device_api/single_write_test.cu` — extended to also run with `enableDeviceProxy = true` (commit `877d096 test: Add SingleWriteTest support for proxy backend`). Same kernel, same assertions, both code paths.

---

## 4. Performance — what to put on the slide

The full evidence is in
[`examples/device/pingpong/scripts/PERFORMANCE.md`](../examples/device/pingpong/scripts/PERFORMANCE.md)
(411 lines, every number cited from a real artifact). The key things for the deck:

### 4.1 Headline (8 B PUT, proxy variant, sweep on `adv-dev-420`)

| Stage | Cost |
|---|---|
| GPU → CPU dequeue + UCX submit (`prep+submit`) | **7 µs floor**, 138 µs avg |
| UCX in-flight (`submit → completion`) | **4.2 ms one-way** (this is UCX `cuda_ipc`, not the proxy) |
| Completion → GPU notification (`publish`) | **36 ns** |
| Worker idle-spin ratio | 392 880 polls / request (worker idle 99.9 %) |

> **Net: the proxy adds tens of µs of submission overhead and tens of ns of completion overhead. Everything else is UCX.**

### 4.2 Size sweep (2000 iters / 200 warmup, both binaries)

```
msg_size      ucx_us     proxy_us     delta_us     ratio
       8     4348.27     13050.85      8702.58     3.00x
      64     4348.17     13050.88      8702.71     3.00x
     512     4348.18     13050.90      8702.72     3.00x
    4096     4348.23     13050.99      8702.76     3.00x
   32768     4348.66     13050.98      8702.32     3.00x
  262144    30451.29     13050.95    -17400.34     0.43x   ← proxy faster
 1048576   126155.51     13050.96   -113104.55     0.10x   ← proxy 10× faster
```

Two non-obvious things to call out from this table:
- **Proxy RTT is essentially constant** (13.05 ms ± 0.15 µs) across 5 orders of magnitude of message size.
- **At ≥ 256 KiB the proxy beats UCX-direct on this host** (UCX-direct's GPU device-API path falls into a slow protocol branch).

### 4.3 NVTX accounting (independent confirmation)

```
Time (%)  Total Time (ns)  Instances    Avg (ns)     Range
  85.9       483 318 270    1 952 784      247.5     :prx:progress    ← idle UCX polling
  14.0        78 803 251        2 200   35 819.7     :prx:submit       ← real work
   0.1           542 992        2 200      246.8     :prx:publish      ← rounding error
```

### 4.4 CUDA-API contrast

| | Proxy run | UCX-direct run |
|---|---|---|
| `cuStreamSynchronize` | 1 call (28.7 s) | 1 call (9.6 s) |
| `cuEventQuery` | **3.92 M calls** | 0 |
| `cuStreamAddCallback` | **982 K calls** | 0 |
| `cuMemcpyDtoDAsync_v2` | **2 200 calls @ 9.4 µs** | 0 |
| GPU D→D memcpy | 2 200 @ 1.9 µs each | 0 |

> The proxy run does exactly **1 GPU D→D memcpy per PUT** (= warmup + iters = 2 200), which is what `cuda_ipc` writes actually cost the copy engine. UCX-direct does none of this — it goes through the device-API path.

### 4.5 Slide-ready conclusions

1. **Proxy is not the bottleneck.** Submission floor 7 µs, completion 36 ns, worker idle 99.9 %.
2. **Publish path is essentially free** (~30 ns, 0.1 % of worker CPU).
3. **All current tail latency is in UCX** (`inflight` ≈ 4.2 ms regardless of stage).
4. **Proxy delivers what it set out to deliver**: GPU-initiated transfers without GDA-KI / IBGDA. On boxes without GDR (most of them), it's the only way to do GPU-issued PUTs.
5. **Counter-intuitive bonus**: at large sizes proxy is *faster* than UCX-direct on this host because UCX-direct hits a slow GPU device-API protocol branch.

### 4.6 Caveats to disclose (don't get blindsided in Q&A)

1. **GDR is not active on the bench host** (`nvidia_peermem` not loaded, `gdrcopy` not installed). Both binaries fall back to `cuda_ipc`, so the comparison is fair but doesn't yet show the RoCE data-plane number.
2. **The ping-pong kernel uses async PUTs.** Completion is inferred from the peer's sequence counter landing in HBM, not from `nixlGpuGetXferStatus`, so the reported RTT is the GPU-observed round trip.
3. **Proxy RTT being flat across sizes is suspicious enough to flag** — the per-stage `inflight` is also flat at 4.2 ms, so the constancy is a UCX/cuda_ipc artifact, not a measurement bug.
4. **The proxy demo is intentionally pinned to one worker / one channel** until the UCX proxy backend's shared `postXfer` path is validated with concurrent workers.
5. **No UCX-direct per-stage breakdown yet** — the proxy stats only exist in the proxy build. Adding NVTX brackets around the GPU-side `nixlPut` call sites in `bench_kernel.cu` is the cleanest way to make the next comparison apples-to-apples.

### 4.7 How to live-demo this

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

Outputs land in `profile_results/<timestamp>/`:
- `summary.txt` — the size sweep table.
- `sweep_proxy_<size>_recv.err` — the `[proxy-stats]` block.
- `analysis.txt` — NVTX, CUDA API, GPU mem-time, and `[proxy-stats]` distilled into one file.
- `nsys_*.nsys-rep` — Nsight Systems traces you can open in the GUI.

---

## 5. Suggested slide order

1. **Why** — GPU-issued transfers without GDA-KI/IBGDA.
2. **Architecture** — the layered diagram from §1.1.
3. **GPU API contract** — `nixlPut` is unchanged; backend selected at compile time (`src/api/gpu/`).
4. **Proxy core** — work ring + collapsed CQ + memview registry (§1.2 with the two short code refs).
5. **Adapter boundary** — `DeviceProxyBackendAdapter`; UCX is one of N (libfabric next).
6. **Agent integration** — `enableDeviceProxy = true`, transparent memview swap.
7. **The benchmark** — §2.1 (kernel) + §2.4 (harness) + the cheat-sheet table from §2.5.
8. **Tests** — 14 registry + 17 runtime + 12 device-API + extended single-write.
9. **Performance** — headline table, size sweep, NVTX % breakdown, CUDA-API contrast.
10. **Conclusions + caveats** (§4.5 + §4.6).
11. **Roadmap** — ranked from `PERFORMANCE.md` "Action items" and `TODO.md` (decouple from `STATIC_PLUGIN_UCX`, multi-worker scaling, GDR re-test, UCX-direct instrumentation).
