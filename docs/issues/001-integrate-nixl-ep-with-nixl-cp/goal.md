# Goal
> 001-integrate-nixl-ep-with-nixl-cp | feature | high-verb

## Problem
The EP maintainer and the NIXL CPU proxy maintainer need to validate that the existing NIXL EP example can execute through the CPU-proxy GPU device backend, not only through the current UCX-direct path. Without this integration, maintainers cannot prove that EP HT and elastic LL workloads work correctly through the proxy path, cannot isolate proxy-specific correctness issues, and cannot later make an apples-to-apples comparison between UCX-direct and CPU-proxy behavior.

## Target
Deliver a correctness-first Phase 1 in which maintainers can run the existing EP workflow with the NIXL CPU-proxy backend while keeping the UCX-direct workflow stable. Observable success means the proxy backend path is configured and exercised, HT correctness passes using a runnable topology or an explicitly added HT-compatible proxy smoke path, the elastic LL suite passes with evidence that its accepted all-RDMA fallback was used under the proxy, and a UCX-direct correctness smoke remains green.

The UCX-direct stability evidence is a separate correctness signal, not part of the deferred UCX-direct versus CPU-proxy comparison artifact. Phase 1 is not gated on proxy throughput, performance tuning, a UCX-direct versus CPU-proxy comparison artifact, proxy multi-worker scaling, full two-node comparison/performance completion, or restoring the LL NVLink/P2P fast path. Those outcomes remain visible follow-ons after the proxy correctness boundary is established.

## Constraints
- G-001: Phase 1 is correctness-first; passing EP HT and elastic LL correctness through the CPU-proxy backend, with the evidence boundaries below, is the acceptance boundary.
- G-002: The existing UCX-direct EP workflow must remain stable for users who build and run EP without the proxy backend, and that stability must be shown by a correctness smoke/regression signal separate from any deferred UCX-direct versus CPU-proxy comparison artifact.
- G-003: The EP Python-facing workflow and module identity should remain stable; Phase 1 should not introduce a redesigned user entry point.
- G-004: Phase 1 targets one CPU proxy worker with N proxy channels, where N covers EP's logical device-side lanes.
- G-005: Proxy multi-worker scaling, channel-to-worker mapping, and UCX worker/QP selection are out of scope for Phase 1.
- G-006: Phase 1 validation must use a runnable EP topology. The known true single-node HT fallback that depends on fewer local ranks, or on exactly eight local ranks with only eight total ranks, is not valid Phase 1 evidence under the current HT test constraints; any single-node evidence must come from an explicit compatible smoke/test change rather than that impossible fallback.
- G-007: A real two-node proxy RDMA run is valuable follow-on evidence and may be the most direct way to satisfy the current HT topology constraints, but a full two-node comparison/performance run is not required before initial Phase 1 completion if a different runnable proxy correctness path is defined.
- G-008: Elastic LL all-RDMA fallback under the proxy is acceptable for Phase 1 only when validation captures an explicit signal that the fallback was used; proxy-side `nixlGetPtr` and NVLink/P2P restoration are deferred.
- G-009: No minimum proxy performance threshold, performance tuning task, or UCX-direct versus CPU-proxy performance artifact gates Phase 1 acceptance.
- G-010: No new third-party dependency is expected for Phase 1; the work should use the existing NIXL GPU Device API proxy backend, CUDA device-linking support, UCX backend/provider, and EP test environment.
- G-011: Phase 1 should consume existing CPU proxy capabilities rather than changing CPU proxy internals; internal proxy scaling is a separate milestone.
- G-012: Architecture/tasks must define explicit evidence for the expected proxy all-RDMA fallback before Phase 1 can be accepted, because the exact signal is still unresolved at the goal stage.
- G-013: Architecture/tasks must define acceptable reduced-size EP configurations before validation if default settings time out under the one-worker proxy model; reduced configurations are correctness evidence only and not performance tuning deliverables.

## Assumptions
- The staged plan in `ep-integ-plan.md` is source material, but Phase 1 acceptance is narrower than the full comparison and optimization plan; any comparison or tuning work in that plan is follow-on unless later artifacts explicitly separate it from Phase 1 acceptance.
- The candidate proxy channel count is N = `max(num_sms / 2, num_local_experts)`, matching the current understanding of HT and LL channel-id usage; architecture should verify this against the current EP code and any override behavior.
- The existing proxy path supports the device operations needed for HT and for elastic LL's all-RDMA fallback.
- Proxy `nixlGetPtr` does not currently restore the LL NVLink/P2P fast path, so Phase 1 treats that behavior as a known fallback rather than a defect.
- One CPU proxy worker may bottleneck EP workloads; if that blocks correctness validation, reduced test sizes or timeout adjustments may be acceptable only when they still prove the intended proxy path and were defined as acceptable correctness evidence before the run.
- UCX-direct stability should be proven with a small correctness smoke or regression check that is independent of any later UCX-direct versus CPU-proxy comparison table or plot.
- Performance data remains useful for follow-on decisions, but it should not be used to reject or complete Phase 1 if the required correctness evidence is otherwise missing or present.

## Phases
### Phase 1: Correctness-first EP-on-proxy
Enable the EP maintainer and NIXL CPU proxy maintainer to run the existing EP workflow through the CPU-proxy backend and validate HT plus elastic LL correctness. This phase keeps the UCX-direct workflow stable with a separate correctness smoke, uses one proxy worker with enough proxy channels for EP's logical lanes, validates HT through a runnable topology or explicit compatible smoke path rather than the known impossible true single-node fallback, and accepts elastic LL's all-RDMA fallback only with an observable fallback signal.

### Phase 1.5: CPU proxy scaling follow-on
After Phase 1 correctness is green, evaluate CPU proxy internal scaling and any UCX-direct versus CPU-proxy performance comparison as distinct follow-on milestones. This includes multiple proxy workers/channels, channel-to-worker mapping, UCX worker/QP selection, transport thread-safety or progress behavior under multiple proxy threads, and performance-oriented tuning.

### Phase 2: Optional LL peer-pointer restoration
Investigate whether proxy-side `nixlGetPtr` can provide device-usable peer pointers for NVLink/P2P-capable peers. Decide whether restoring the LL fast path is worth the complexity after Phase 1 behavior and later performance evidence are understood.
