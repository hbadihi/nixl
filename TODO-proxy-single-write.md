# Proxy SingleWriteTest — Next Steps

## Currently Working
- `SingleWorkerPut` passes for BLOCK, WARP, and THREAD levels via the proxy CPU path.

## Skipped (needs implementation)

### MultipleWorkersPut
- Multiple proxy workers/channels are not yet supported end-to-end.
- The proxy enqueue design currently assumes a single producer per channel;
  concurrent GPU threads writing to the same work ring can race.
- Next steps:
  1. Add per-channel locking or single-producer enforcement in the work ring.
  2. Support multi-channel `prepMemView` so each worker gets its own memview pair.
  3. Un-skip the test and validate with `numWorkers > 1`.

### SingleWorkerPutGap (`nixlGetPtr`)
- `nixlGetPtr` is not implemented for the proxy backend.
- The proxy device API currently only exposes `nixlPut`; gap/scatter patterns
  that rely on `nixlGetPtr` for address arithmetic are unsupported.
- Next steps:
  1. Implement `nixlGetPtr` in `src/api/gpu/proxy/nixl_device_impl.cuh`.
  2. Wire it through `ProxyDeviceContextData` (or resolve addresses host-side).
  3. Un-skip the test.

## Build Notes
- `gtest_proxy` requires `-Dstatic_plugins=UCX` in the Meson configuration so
  that `STATIC_PLUGIN_UCX` is defined and `createProxyRuntime` is compiled in.
- The `NIXL_GPU_DEVICE_BACKEND_PROXY` define must be passed via both `cpp_args`
  and `cuda_args` in the Meson build file (already done).

## Bug Fixed in This Change
- `nixlUcxProxyBackend::loadRemoteConnInfo` was calling
  `engine_->loadRemoteConnInfo()` on the shared UCX engine that already had the
  connection loaded by the agent path, causing `NIXL_ERR_INVALID_PARAM`.
  Fixed by guarding with `checkConn()` first.
