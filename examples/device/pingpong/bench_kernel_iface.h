#pragma once

// Host-safe interface header: no nixl_device.cuh, no CUDA device headers.
// Include this from .cpp files; include bench_kernel.cuh from .cu files.

#include "nixl.h"
#include <cuda_runtime.h>
#include <cstdint>

// ---- GPU-side context passed directly into the kernel -------------------------
// All pointers must be device-accessible (cudaMalloc'd).

struct gpu_bench_ctx {
    nixlMemViewH local_mvh;  // view of local send_buf
    nixlMemViewH remote_mvh; // view of peer's recv_buf
    uint8_t     *send_buf;   // device pointer, buf_size bytes
    uint8_t     *recv_buf;   // device pointer, buf_size bytes
    size_t       msg_size;   // payload bytes (counter NOT included)
    uint64_t     num_iters;
    uint64_t     warmup_iters;
    bool         is_sender;
};

// ---- Launch wrappers ----------------------------------------------------------
// d_elapsed: device pointer to a single uint64_t; receives clock64 ticks for
//            the timed phase (warmup excluded).  Only meaningful for is_sender.

void launch_pingpong_thread(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);
void launch_pingpong_warp  (gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
// Host-callable thin wrappers around nixlProxyPublishContext/ClearContext.
// Defined in bench_kernel.cu so the host .cpp file does not need to include
// nixl_device_proxy.cuh (which references CUDA device builtins).
//
// proxy_ctx must be the value returned by nixlAgent::getProxyDeviceContext().
cudaError_t bench_proxy_publish_context(void *proxy_ctx);
cudaError_t bench_proxy_clear_context();
#endif
