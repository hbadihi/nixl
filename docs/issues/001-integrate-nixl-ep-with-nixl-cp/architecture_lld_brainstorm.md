---
issue-id: 001-integrate-nixl-ep-with-nixl-cp
stage: discuss_architecture_lld
timestamp: 2026-06-10T10:10:46Z
---

# Architecture LLD Brainstorm

## Decisions

1. Keep the LLD inside the existing EP build, host runtime, and validation tree.
   - `examples/device/ep/meson.build` owns selected-backend build wiring.
   - `examples/device/ep/csrc/nixl_ep.cpp` owns proxy enablement in the existing `Buffer` lifecycle.
   - `examples/device/ep/csrc/kernels/proxy_publish.cu` should be the only new proxy-specific CUDA translation unit.
   - `examples/device/ep/tests/` owns the explicit HT-compatible proxy smoke and validation evidence checks.
   - No new Python module, public `Buffer` API, service process, runtime backend selector, or proxy-specific HT/LL kernel fork is part of Phase 1.

2. Extend the EP Meson gate without changing module identity.
   - Replace the current UCX-only hard error with an allow-list for `ucx` and `proxy`; keep `none` and missing dependency skip behavior.
   - In proxy builds, add the existing proxy include/link pieces (`nixl_gpu_proxy_inc_dirs`, `nixl_gpu_proxy_lib`, and existing GPU Device API compile args).
   - Compile `csrc/kernels/proxy_publish.cu` only when `gpu_device_api_backend=proxy`.
   - UCX-direct builds keep the same `nixl_ep` Python-facing workflow and should not pick up proxy publish code.

3. Add a small host-callable proxy publish/clear seam.
   - The CUDA TU should expose wrappers with signatures equivalent to:
     - `cudaError_t nixl_ep_proxy_publish_context(void *ctx)`
     - `cudaError_t nixl_ep_proxy_clear_context()`
   - `nixl_ep.cpp` passes `nixlAgent::getProxyDeviceContext()` into the publish wrapper after `createBackend("UCX", ...)`.
   - A null proxy context or non-`cudaSuccess` publish result is setup failure.
   - Clear runs before agent/proxy teardown; clear failure should be reported as teardown warning unless the local project pattern requires fatal teardown.

4. Configure proxy lifecycle in `Buffer::_nixl_agent_init()` under the proxy backend macro.
   - Set `cfg.enableDeviceProxy = true`.
   - Set `cfg.proxyWorkerCount = 1`.
   - Set `cfg.proxyChannelCount = required_proxy_channels`.
   - Keep UCX `init_params["num_workers"] = "1"` for Phase 1.
   - Do not modify UCX worker/QP routing, proxy channel-to-worker mapping, or CPU proxy internals in Phase 1.

5. Derive required proxy channel count from the existing allocation-time lane ceiling.
   - `required_proxy_channels` is the `max_experts_per_rank` / `num_experts_per_rank` value passed through `Buffer::update_memory_buffers(...)`.
   - HT already passes `num_qps_per_rank = max(num_sms / 2, LL compatibility lanes)` into `update_memory_buffers`.
   - Elastic LL passes `args.num_experts_per_rank` into the same API.
   - A proxy-only `NIXL_EP_PROXY_CHANNELS` override may raise the channel count, but any value below `required_proxy_channels` is invalid setup and must fail before kernels run.
   - Do not reuse `NIXL_EP_NUM_CHANNELS` for proxy work rings; today it configures UCX-direct `ucx_num_device_channels`, which is a different concept.

6. Preserve existing memview and device-operation contracts.
   - `_nixl_ep_memory_views_create()` and `_nixl_ep_memory_views_destroy()` remain the memview lifecycle seam.
   - `gpu_nixl_ctx`, memview handles, `nixlProxySubmission`, and proxy memview registry behavior remain unchanged.
   - HT and LL kernels continue calling backend-agnostic `nixlPut`, `nixlAtomicAdd`, and `nixlGetPtr` wrappers.
   - Proxy `nixlGetPtr == nullptr` remains the accepted Phase 1 all-RDMA fallback; peer-pointer restoration is deferred.

7. Use the existing error model.
   - Host setup failures throw `std::runtime_error` or trip existing host assertions according to surrounding EP style.
   - Device failures remain `nixl_status_t` results checked by existing device assertions.
   - Under-provisioned proxy channels, missing proxy context, invalid proxy-channel override, and publish failure are setup errors, not accepted validation outcomes.
   - Transport or proxy submit/completion failures are not transparently retried as idempotent success, especially for atomics; they should be validation-visible failure or inconclusive evidence.

8. Define the Phase 1 ordering contract narrowly.
   - Per-channel enqueue order is the only ordering contract Phase 1 should rely on.
   - Data PUT and follow-up flag/atomic operations that depend on ordering must remain on the same channel.
   - No cross-channel ordering or multi-worker routing guarantee is introduced in Phase 1.

9. Add structured correctness evidence as validation output, not as production API.
   - The accepted evidence surface is `ep_proxy_evidence_v1`, emitted by EP validation code or tests.
   - The evidence record should include backend selection, rank, worker count, channel count, required channel count, proxy context published, proxy activity observed, LL all-RDMA fallback observed or not applicable, correctness result, and classification.
   - Proxy activity evidence must prove submitted work during the HT or LL run, not only proxy runtime creation.
   - LL fallback evidence must be explicit and EP-visible; correctness-only inference from `nixlGetPtr` behavior is not enough.
   - Debug-log archaeology is not accepted evidence. Lightweight deterministic logs or counters may back the structured record.

10. Add a dedicated HT-compatible proxy smoke under `examples/device/ep/tests/`.
    - The smoke is separate from the full two-node `test_ht.py` path.
    - It must prove HT correctness and CPU-proxy activity together.
    - The existing true single-node `test_ht.py` fallback remains rejected or inconclusive under current rank constraints.
    - A full two-node HT RDMA run remains valuable follow-on evidence, not the only initial Phase 1 correctness route.

11. Keep test seams layered.
    - Existing device API proxy tests remain the lower-level unit/integration seam for proxy publish, clear, enqueue, and channel behavior.
    - EP tests validate selected backend build/import, proxy lifecycle, channel coverage, HT proxy smoke, elastic LL fallback evidence, and independent UCX-direct correctness.
    - Validation classification lives close to EP tests unless implementation finds a strong reason for a small helper; it should not become a public runtime API.

12. Defer Phase 1.5 and Phase 2 contracts.
    - Multi-worker proxy scaling, `submission.channel_id % num_workers`, UCX worker/QP selection, and UCX multi-thread/progress validation are Phase 1.5.
    - Proxy-side `nixlGetPtr` peer-pointer restoration is Phase 2 and requires rank, memview, bounds, lifetime, and authorization boundaries before returning device-usable pointers.

## Open Questions

1. What reduced-size validation floor can count as correctness evidence if one proxy worker times out?
   - This remains unresolved by the LLD brainstorm.
   - Any reduced-size run must be defined before validation and must still prove the proxy path, channel coverage, and LL fallback evidence.
   - Ad hoc downsizing remains inconclusive.

2. How much proxy-runtime instrumentation already exists for work-submission counters?
   - The LLD should prefer existing counters or deterministic low-cost logging if present.
   - If no stable runtime counter exists, tasks should add the smallest correctness-only signal needed to back `ep_proxy_evidence_v1`.

3. Should under-provisioned proxy channels use `std::runtime_error` or `EP_HOST_ASSERT`?
   - The panel recommends matching the surrounding EP setup-failure style.
   - The required behavior is more important than the exact mechanism: fail before kernels run with an actionable reason.

## Worked Shapes

### Build Shape

1. Configure `build-ucx` with `-Dgpu_device_api_backend=ucx`.
2. Configure `build-proxy` with `-Dgpu_device_api_backend=proxy`.
3. `examples/device/ep/meson.build` allows both backends.
4. Proxy builds link the existing proxy device library and compile `proxy_publish.cu`.
5. UCX-direct builds keep the current module and import behavior.

### Runtime Shape

1. Python calls `Buffer.update_memory_buffers(num_ranks, num_experts_per_rank, ...)`.
2. C++ stores that value as the EP allocation-time lane ceiling.
3. In proxy builds, `_nixl_agent_init()` sets proxy config with one worker and `proxyChannelCount >= required_proxy_channels`.
4. The UCX backend is created with `num_workers=1`.
5. EP obtains the proxy device context from the agent and publishes it through the CUDA wrapper.
6. Existing memview preparation runs unchanged.
7. HT/LL kernels issue backend-agnostic GPU Device API calls.
8. The CPU proxy runtime consumes channelized work and submits UCX operations.
9. Teardown clears proxy context before releasing agent/proxy resources.

### Channel Override Shape

1. Default: `proxyChannelCount = required_proxy_channels`.
2. Optional override: `NIXL_EP_PROXY_CHANNELS`.
3. If override is absent, use the default.
4. If override is present and invalid or below required, fail setup.
5. If override is present and above required, accept it as extra channel capacity.
6. `NIXL_EP_NUM_CHANNELS` remains UCX-direct `ucx_num_device_channels` behavior.

### Evidence Shape

```yaml
ep_proxy_evidence_v1:
  backend: proxy | ucx
  rank: <rank>
  proxy_worker_count: 1
  proxy_channel_count: <configured>
  required_proxy_channels: <derived>
  proxy_context_published: true | false
  proxy_activity_observed: true | false
  ll_all_rdma_fallback_observed: true | false | not_applicable
  correctness: pass | fail | not_run
  classification: accepted | failed | inconclusive
  reason: "<short actionable reason>"
```

Accepted HT evidence requires `backend=proxy`, published proxy context, proxy activity, correctness pass, and accepted smoke/topology metadata.

Accepted elastic LL evidence requires `backend=proxy`, proxy activity, explicit all-RDMA fallback observed, and correctness pass.

UCX-direct smoke evidence is separate and should not be represented as proxy acceptance.

### Negative-Test Shape

1. Proxy build with missing proxy context: setup failure.
2. Proxy build with `NIXL_EP_PROXY_CHANNELS < required_proxy_channels`: setup failure.
3. HT smoke passes correctness but has no proxy activity: inconclusive.
4. Elastic LL passes correctness but lacks fallback signal: inconclusive.
5. UCX-direct smoke fails after proxy integration: Phase 1 blocked until fixed.
6. One-worker timeout without pre-approved reduced-size criteria: inconclusive or validation-blocked.

### Deferred M2 Shape

1. Change proxy worker count to N.
2. Change UCX `num_workers` to N.
3. Plumb worker id through UCX submit paths.
4. Map proxy channel to worker explicitly.
5. Validate UCX multi-thread safety and progress behavior.
6. Only do this after Phase 1 correctness is accepted.

## Panel

### system-arch

- Reaffirmed the HLD boundary: existing EP rank process, build-time backend selection, one proxy worker, N channels, no new service.
- Highlighted per-channel ordering, non-idempotent atomic retry risk, and validation evidence as cross-component contracts.
- Confidence: low before human convergence because evidence and external UCX assumptions needed pinning.

### sw-arch

- Proposed the concrete module/file shape: Meson gate, `proxy_publish.cu`, guarded `nixl_ep.cpp` lifecycle, no public Python API change.
- Recommended `NIXL_EP_PROXY_CHANNELS` as a proxy-only override and `ep_proxy_evidence_v1` as a validation artifact rather than production API.
- Confidence: medium before human convergence.

### sw-dev

- Rated build wiring and existing memview reuse as clear.
- Flagged channel sizing, LL fallback evidence, proxy activity evidence, and HT smoke placement as hidden-cost or needs-clarification seams.
- Confidence: low before human convergence because channel and evidence contracts had to be pinned.

## Human Convergence

The human accepted the following LLD convergence choices during Beat 2.5:

1. `cfg.proxyChannelCount` should be derived from the allocation-time lane ceiling passed through `update_memory_buffers(...)`, with `NIXL_EP_PROXY_CHANNELS` as a validated proxy-only override that must be greater than or equal to the required channel count.
2. The explicit HT-compatible proxy smoke should live as a small dedicated test path under `examples/device/ep/tests/`, separate from the full two-node `test_ht.py` path.
3. Phase 1 should require structured `ep_proxy_evidence_v1` output from EP/tests, backed by lightweight runtime counters or deterministic logs for backend selection, proxy activity, and LL all-RDMA fallback. Debug-log archaeology is not accepted evidence.
