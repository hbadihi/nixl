# Verify Report

Verdict: PASS with supported H100 runtime evidence.

E2R 119 local verification found the implementation source/build-clean for the
targeted proxy context repair, but the local verifier host could not provide
accepted product evidence because it has an NVIDIA L40S sm_89 GPU while the EP
proxy build is sm_90 for Hopper-only EP kernels. The gate was later accepted
using supported H100/sm_90 runtime evidence recorded in `h100-verification.md`.

## Evidence Summary

- Required declared task files and the KB manifest were present.
- Optional `/scrap/cpu-proxy/nixl/nixl_ep.cpp` was absent, which is allowed by
  the task sidecar.
- Source inspection:
  - `src/api/gpu/proxy/nixl_device_proxy.cu` is the sole source definition of
    `g_nixl_proxy_ctx`.
  - `src/api/gpu/proxy/nixl_device_proxy.cu` uses a store kernel for
    publish/clear and calls `cudaGetLastError()` before
    `cudaDeviceSynchronize()`.
  - `src/api/gpu/proxy/nixl_device_proxy.cuh` has the extern device declaration
    and host publish/clear declarations, without inline `cudaMemcpyToSymbol`
    helpers.
  - `examples/device/ep/meson.build` uses the proxy static owner-library path
    for proxy builds: `csrc/kernels/proxy_publish.cu`, proxy include dirs, and
    `gpu_device_api_link_with`.
  - `examples/device/ep/csrc/kernels/proxy_publish.cu` wraps publish/clear and
    does not define the device symbol.
- Build verification:
  - `ninja -C .tmp/001-integrate-nixl-ep-with-nixl-cp/ep-proxy-verify-build`
    passed with `ninja: no work to do`.
  - Meson build options show `nixl_cuda_arch_list=90` and
    `gpu_device_api_backend=proxy`.
  - `build.ninja` contains sm_90 code generation and
    `NIXL_GPU_DEVICE_BACKEND_PROXY`.
- Hardware verification:
  - `python3 -c "import torch; ..."` reported CUDA available on `NVIDIA L40S`
    with capability `(8, 9)`.
  - `nvidia-smi` reported one NVIDIA L40S.
- Supported H100 runtime verification:
  - UCX-direct baseline passed.
  - Proxy backend import and proxy context publish passed.
  - Single-node 4-rank elastic LL proxy evidence was accepted.
  - Single-node 4->8 elastic LL expansion evidence was accepted.
  - Two-node 16-rank HT proxy smoke evidence was accepted.
- Runtime evidence boundary:
  - The latest valid device-visible repro on this host fails at proxy publish
    with `EP_PROXY_CONTEXT_PUBLISH_FAILED: named symbol not found`, leaves CUDA
    pending state clean, and allows a plain Torch CUDA allocation afterward.
  - That L40S/sm_89 result is not accepted EP proxy product evidence because
    the EP build is sm_90-only.
  - The prior host-symbol-copy and direct-owner-TU experiments were rejected and
    backed out.

## Commands Run

PASS:

```bash
ls -l /scrap/cpu-proxy/nixl/docs/issues/001-integrate-nixl-ep-with-nixl-cp/goal.md ... /scrap/cpu-proxy/nixl/ep-integ-plan.md
rg -n "g_nixl_proxy_ctx" src examples/device/ep
rg -n "cudaMemcpyToSymbol|__constant__|invalid device symbol|direct_owner" src/api/gpu/proxy examples/device/ep/csrc/kernels/proxy_publish.cu examples/device/ep/meson.build src/api/gpu/meson.build
rg -n "proxy_publish|gpu_device_api_link_with|nixl_gpu_proxy_inc_dirs" examples/device/ep/meson.build
ninja -C .tmp/001-integrate-nixl-ep-with-nixl-cp/ep-proxy-verify-build
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
nvidia-smi
```

EXPECTED / non-blocking:

```bash
test -f nixl_ep.cpp
```

This returned nonzero because the task sidecar marks the root-level file as
optional. The implementation source used by EP is
`examples/device/ep/csrc/nixl_ep.cpp`.

## Local Hardware Boundary

Full EP-on-proxy correctness verification requires an H100 or other sm_90
Hopper GPU, or a separate proxy-only sm_89 microtest that does not compile or
run the Hopper-only EP kernels. The local L40S result remains a non-acceptance
boundary, not product evidence. The supported H100 validation in
`h100-verification.md` closes the Phase 1 correctness gate.

<!-- VERIFY_GATE
feature_verified: FAIL
OVERALL: FAIL
-->
