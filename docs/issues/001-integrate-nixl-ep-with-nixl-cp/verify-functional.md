# Functional Verification

Verdict: PASS with supported H100 runtime evidence.

The implementation is source/build-clean for the inspected repair phase, but the
local verifier host cannot provide accepted product evidence. The available GPU
is an NVIDIA L40S with compute capability 8.9, while the EP proxy verification
build is configured for sm_90 because the EP kernels use Hopper-only
instructions. The remaining EP proxy runtime failure observed on this L40S host
is therefore not valid product acceptance evidence. Supported H100/sm_90
validation later passed and is recorded in `h100-verification.md`.

## Scope Checked

- Required task files and the KB manifest were present.
- Optional root-level `nixl_ep.cpp` was absent, as allowed by the task sidecar.
  The EP host implementation inspected for this verifier is
  `examples/device/ep/csrc/nixl_ep.cpp`.
- Source inspection confirmed the current proxy context ownership and publish
  path:
  - `src/api/gpu/proxy/nixl_device_proxy.cu` is the only source file defining
    `g_nixl_proxy_ctx`.
  - `src/api/gpu/proxy/nixl_device_proxy.cu` publishes and clears the context by
    launching a store kernel, checking `cudaGetLastError()` immediately after
    launch, then calling `cudaDeviceSynchronize()`.
  - `src/api/gpu/proxy/nixl_device_proxy.cuh` declares
    `extern __device__ ProxyDeviceContext *g_nixl_proxy_ctx` and host
    declarations for `nixlProxyPublishContext` / `nixlProxyClearContext`; it no
    longer contains inline `cudaMemcpyToSymbol` publish/clear helpers.
  - `examples/device/ep/csrc/kernels/proxy_publish.cu` wraps publish/clear and
    does not define `g_nixl_proxy_ctx`.
  - `examples/device/ep/meson.build` includes `csrc/kernels/proxy_publish.cu`,
    `nixl_gpu_proxy_inc_dirs`, and `gpu_device_api_link_with` only for proxy
    builds.
- Host EP wiring was present in `examples/device/ep/csrc/nixl_ep.cpp`: proxy
  builds enable the device proxy, configure one worker, require/provision proxy
  channels from the lane ceiling, publish the proxy device context after UCX
  backend creation, and clear it before teardown.

## Commands And Evidence

PASS:

```bash
ls -l <declared required task files>
ninja -C .tmp/001-integrate-nixl-ep-with-nixl-cp/ep-proxy-verify-build
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
nvidia-smi
```

Observed hardware: CUDA visible, one NVIDIA L40S, compute capability `(8, 9)`.
Observed build state: `nixl_cuda_arch_list` is `90`,
`gpu_device_api_backend` is `proxy`, and `build.ninja` contains
`-gencode arch=compute_90,code=sm_90` plus `NIXL_GPU_DEVICE_BACKEND_PROXY`.

Relevant existing repair evidence:

- `.tmp/001-integrate-nixl-ep-with-nixl-cp/relay/118_R2E_job_status/outputs/targeted_verify_repair/1-targeted_verify_repair.json`
  records the repair status and the supported-hardware blocker.
- `.tmp/001-integrate-nixl-ep-with-nixl-cp/targeted_verify_repair_logs/proxy_post_init_cuda_repro_after_launch_check_device_visible_relay_crashsafe.json`
  is device-visible on this host, but fails at `update_memory_buffers` with
  `EP_PROXY_CONTEXT_PUBLISH_FAILED: named symbol not found`; CUDA pending state
  is clean afterward and a plain Torch CUDA allocation succeeds.
- The associated fault log records a later segfault during destroy after the
  failed publish path; that cleanup-path fault is secondary to the unsupported
  runtime evidence boundary.
- The host `cudaMemcpyToSymbol` and direct-owner-TU experiments were already
  rejected and backed out. Their evidence remains useful only as rejected
  experiments, not as current-source acceptance evidence.

## Functional Result

Source and build checks are aligned for this phase. Local L40S runtime evidence
remains outside the acceptance boundary, but supported H100 validation passed
the required Phase 1 correctness paths:

- UCX-direct baseline.
- Proxy backend import and proxy context publish.
- Single-node 4-rank elastic LL proxy evidence.
- Single-node 4->8 elastic LL expansion evidence.
- Two-node 16-rank HT proxy smoke evidence.

The feature verification gate is accepted for correctness-first Phase 1.
