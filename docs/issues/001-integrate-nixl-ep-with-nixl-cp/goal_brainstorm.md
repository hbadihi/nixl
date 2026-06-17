---
issue-id: 001-integrate-nixl-ep-with-nixl-cp
stage: discuss_goal
timestamp: 2026-06-09T12:05:36Z
---

# Goal Brainstorm: Integrate NIXL EP With NIXL CPU Proxy

## Decisions

1. Phase 1 is a correctness-first integration of the existing NIXL EP example with the NIXL CPU Proxy GPU device backend.

   The goal is for the EP maintainer and NIXL CPU proxy maintainer to validate that EP workloads can execute through the proxy path without changing the existing UCX-direct user workflow. Phase 1 acceptance is not gated on a performance comparison artifact.

2. Phase 1 targets one CPU proxy worker with N proxy channels.

   N must cover EP's logical device-side lanes. The worker count and channel count are separate knobs: Phase 1 should provide enough channels for correctness while keeping a single proxy worker to avoid taking on proxy scaling and UCX multi-worker thread-safety work in the first milestone.

3. Phase 1 validation starts with single-node multi-process execution.

   A real two-node proxy RDMA run is valuable and should follow, but it is not required before initial Phase 1 completion. Single-node validation must still prove that the proxy backend path is configured and exercised.

4. Elastic LL all-RDMA fallback under proxy is acceptable for Phase 1.

   Restoring NVLink/P2P behavior through proxy-side `nixlGetPtr` is deferred. The Phase 1 goal should call this out directly so the implementation does not drift into peer-pointer restoration.

5. No minimum proxy performance threshold gates Phase 1.

   Correctness through the proxy path is the acceptance criterion. Performance measurement, UCX-direct versus CPU-proxy comparison artifacts, and proxy tuning belong to follow-on work after correctness is established.

6. Phase 1.5 is a distinct follow-on for CPU proxy internal scaling.

   Multi-worker proxy infrastructure, channel-to-worker mapping, and UCX worker/QP selection should be treated as a separate milestone after Phase 1. That milestone can focus on multiple workers/channels and the transport thread-safety implications.

7. Phase 2 remains optional and later.

   Proxy `nixlGetPtr` / NVLink P2P restoration should be considered only after Phase 1 correctness and baseline behavior are understood.

## Open Questions

1. What exact observable evidence should prove the elastic LL path used the expected proxy all-RDMA fallback?

   The goal accepts the fallback, but architecture and validation should define the signal: logs, counters, disabled CUDA IPC/NVLink path, proxy worker activity, or another explicit marker.

2. Which reduced-size EP configurations are acceptable if default settings timeout under the one-worker Phase 1 proxy model?

   The goal allows correctness-first validation, but later stages should define how to avoid confusing expected M1 bottlenecks with correctness failures.

3. What is the precise definition of "N proxy channels" for each EP mode?

   The reference plan suggests `N = num_qps_per_rank = max(num_sms / 2, num_local_experts)`. Architecture should verify this against current HT and LL channel-id usage and document any override behavior.

4. What follow-on artifact should capture the UCX-direct versus CPU-proxy performance comparison?

   The comparison is explicitly outside Phase 1 acceptance, but it should remain visible as a later task so the feature does not lose the original apples-to-apples motivation.

## Worked Shapes

### Phase 1 Shape

- Build EP in proxy mode by allowing `gpu_device_api_backend=proxy` for the EP extension.
- Keep the Python module name and existing EP workflow stable.
- Enable CPU proxy on the EP agent only under the proxy backend build.
- Publish the proxy device context before EP kernels issue device-side NIXL operations.
- Clear the proxy device context during teardown.
- Configure `proxyWorkerCount = 1`.
- Configure `proxyChannelCount = N`, where N covers the channel ids EP kernels may enqueue.
- Run single-node multi-process validation first.
- Treat HT correctness and elastic LL correctness under proxy fallback as the Phase 1 gate.

### Out Of Scope For Phase 1

- No proxy multi-worker scaling.
- No UCX worker/QP routing changes for proxy submissions.
- No proxy-side `nixlGetPtr` / NVLink P2P restoration.
- No performance threshold.
- No required two-node completion gate.
- No required UCX-direct versus proxy performance artifact.
- No redesign of EP's public Python module or user-facing workflow.

### Phase 1.5 Shape

- Revisit CPU proxy internal scaling after Phase 1 correctness is green.
- Decide how EP channel ids map to proxy workers.
- Decide whether proxy workers should map to UCX workers/QPs.
- Validate UCX thread-safety and progress behavior under multiple proxy threads.
- Add targeted tests for ordering and completion semantics under multiple workers.

### Phase 2 Shape

- Investigate whether proxy `nixlGetPtr` can return device-usable peer pointers for NVLink/P2P-capable peers.
- Decide whether restoring LL fast-path behavior is worth the complexity based on Phase 1 behavior and later performance measurements.

## Panel

- product: Identified the acceptance risk around owner, success signal, topology, and performance artifact boundaries.
- system-arch: Confirmed Phase 1 should stay in-process with the existing EP runtime and use separate UCX/proxy build configurations.
- sw-arch: Pushed the goal toward measurable correctness, explicit non-goals, and separate Phase 1 / 1.5 / 2 boundaries.
- sw-dev: Highlighted validation ambiguity around timeouts, fallback evidence, and proving the intended proxy path was exercised.

## Elicitation Summary

- Phase 1 acceptance is correctness-only; the performance artifact is a separate step.
- Phase 1 validates single-node multi-process first; two-node proxy RDMA follows later.
- The named owners are the EP maintainer and NIXL CPU proxy maintainer.
- Elastic LL all-RDMA fallback is acceptable for Phase 1.
- Correctness is enough for Phase 1; no minimum performance threshold is required.
