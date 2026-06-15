#pragma once

// Windowed one-directional streaming BANDWIDTH benchmark (companion to the
// ping-pong LATENCY bench in simple_bench.h).
//
// Latency vs bandwidth — why a separate kernel:
//   * simple_bench.h keeps exactly ONE put outstanding (issue -> wait reply ->
//     issue ...). That measures round-trip latency, NOT throughput.
//   * This kernel keeps a WINDOW of W puts in flight in ONE direction (sender
//     streams, receiver is a passive RDMA target) and measures sustained
//     bytes/second once the pipeline is full. That is bandwidth / message rate.
//
// Windowing scheme (per issuing warp/thread):
//   prologue : post W puts (slots 0..W-1), each with its own request handle.
//   steady   : for i in [0, warmup+iters): drain slot[i%W] (poll to terminal),
//              then post a fresh put into slot[i%W]. This keeps EXACTLY W puts
//              outstanding; one completes per one posted (credit-based pipeline).
//   timing   : clock64() brackets the `iters` steady iterations after warmup.
//              Each timed iteration completes one put => bytes = iters*msg_size.
//   epilogue : drain the remaining W (not timed).
//
// Parallelism (to probe "parallel SM doorbells" vs "single CPU worker"):
//   The grid launches `warps` independent streams (one warp/thread per block).
//   Each stream uses channel = warp_id % num_channels:
//     * direct (GDAKI): num_channels = RC_GDA channels  -> distinct QPs/doorbells
//       ring in parallel across SMs.
//     * proxy: num_channels = 1 -> every stream funnels into the SINGLE host
//       work ring drained by the SINGLE CPU worker (the hypothesised cap).
//   Aggregate bytes = warps * iters * msg_size; wall = max per-warp clock delta.
//
// Host-safe header (no CUDA device headers). Include stream_kernel.cu's twin
// from .cu code only.

#include "bench_kernel_iface.h" // gpu_bench_op, nixlMemViewH, cuda_runtime.h
#include <cstddef>
#include <cstdint>

// Defined (with a body) in the backend GPU headers; the bench struct below only
// holds a POINTER to it, so a forward declaration keeps this host-safe header
// free of the backend-specific include-path differences (the .cu pulls in the
// full definition transitively via nixl_device.cuh).
struct nixlGpuXferStatusH;

struct stream_bench_ctx {
    nixlMemViewH local_mvh;   // view of local send_buf  (source)
    nixlMemViewH remote_mvh;  // view of peer's recv_buf (destination)
    uint8_t     *send_buf;    // device pointer
    uint8_t     *recv_buf;    // device pointer
    size_t       msg_size;    // payload bytes per put
    gpu_bench_op op;          // Put (data BW) or AtomicFlag (pure message rate)
    uint64_t     num_iters;   // timed posts per warp
    uint64_t     warmup_iters;// untimed posts per warp (pipeline fill / ramp)
    uint32_t     window;      // W: outstanding puts per warp
    uint32_t     num_warps;   // number of independent streams (== gridDim.x)
    uint32_t     num_channels;// channel fan-out (direct: QPs; proxy: 1)
    // Per-warp request-handle ring: num_warps * window entries in device global
    // memory. Kept off the stack so large W does not blow up local memory.
    nixlGpuXferStatusH *reqs;
};

// d_elapsed: device array of num_warps uint64_t; each warp writes its clock64
//            tick delta for the timed region. Host takes the max as wall time.
void launch_stream_thread(stream_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);
void launch_stream_warp  (stream_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);

// sizeof(nixlGpuXferStatusH) — exposed so the host can size the request ring
// without needing the backend-specific include path. Defined in stream_kernel.cu.
size_t stream_xfer_status_bytes();

// bench_proxy_publish_context / bench_proxy_clear_context are declared in
// bench_kernel_iface.h (proxy build) and DEFINED in stream_kernel.cu so the
// reused BenchContext (bench_host.cpp) links against this binary's kernel TU.
