# Architecture Drift Check

Verdict: PASS with supported H100 runtime evidence, with no blocking
implementation drift in the inspected proxy repair path.

## Contract Checked

The declared goal and use cases require Phase 1 correctness evidence for EP HT
and elastic LL through the CPU-proxy backend, plus independent UCX-direct
stability evidence. Build success alone is not acceptance evidence. The LLD
also requires proxy build/link wiring, owner-aware context publish/clear,
one-worker proxy configuration with enough channels, deterministic proxy
activity evidence, and explicit LL all-RDMA fallback evidence.

## Source Alignment

- `g_nixl_proxy_ctx` has one source definition:
  `src/api/gpu/proxy/nixl_device_proxy.cu`.
- The proxy header declares the device symbol extern and exposes host
  publish/clear declarations. The rejected inline `cudaMemcpyToSymbol` path is
  not present in the header.
- The publish/clear implementation is the store-kernel path with launch-error
  attribution before synchronization.
- EP proxy builds compile `examples/device/ep/csrc/kernels/proxy_publish.cu`
  and link the static proxy owner library through `gpu_device_api_link_with`.
- EP host lifecycle code enables proxy mode, provisions channels, publishes the
  proxy context after backend setup, clears it on teardown, and clears it if
  initialization fails after publish.
- The proxy verify build is configured as `gpu_device_api_backend=proxy` and
  `nixl_cuda_arch_list=90`, and `ninja` reports the tree is up to date.

## Drift / Gaps

- Local validation boundary: the verifier host is NVIDIA L40S sm_89, but the EP
  proxy build is sm_90 because the EP kernels require Hopper-only instructions.
  Therefore the local L40S runtime failure cannot be accepted as product
  evidence for or against the feature.
- Missing acceptance evidence: no supported sm_90/Hopper EP proxy runtime run is
  available for this stop, and no separate sm_89 proxy-only microtest exists to
  isolate publish mechanics from EP kernels.
- Non-blocking source-comment drift: `src/api/gpu/meson.build:16-18` still says
  `ProxyRuntime publishes via cudaMemcpyToSymbol`. Current code publishes with
  the store kernel in `src/api/gpu/proxy/nixl_device_proxy.cu`. The comment
  should be corrected in a source-editing stop, but it does not change the
  current binary behavior.

## Architecture Verdict

The inspected implementation shape matches the intended proxy ownership and
build architecture. The gate still fails because the required accepted runtime
evidence is missing on supported hardware, not because this verifier found a
current blocking source architecture defect.
