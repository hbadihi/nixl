> Archived by workflow before rewriting the canonical artifact.
> Replacement: goal.md
> Reason: R3 fixer round 1 for r3_review will rewrite the canonical artifact based on review findings.

# Goal
> 001-integrate-nixl-ep-with-nixl-cp | feature | high-verb

## Problem
The EP maintainer and the NIXL CPU proxy maintainer need to validate that the existing NIXL EP example can execute through the CPU-proxy GPU device backend, not only through the current UCX-direct path. Without this integration, maintainers cannot prove that EP HT and elastic LL workloads work correctly through the proxy path, cannot isolate proxy-specific correctness issues, and cannot later make an apples-to-apples comparison between UCX-direct and CPU-proxy behavior.

## Target
Deliver a correctness-first Phase 1 in which maintainers can run the existing EP workflow with the NIXL CPU-proxy backend while keeping the UCX-direct workflow stable. Observable success means the proxy backend path is configured and exercised, HT correctness passes, and the elastic LL suite passes with its accepted all-RDMA fallback under the proxy.

Phase 1 is not gated on proxy throughput, a UCX-direct versus CPU-proxy comparison artifact, proxy multi-worker scaling, two-node completion, or restoring the LL NVLink/P2P fast path. Those outcomes remain visible follow-ons after the proxy correctness boundary is established.

## Constraints
- G-001: Phase 1 is correctness-first; passing EP HT and elastic LL correctness through the CPU-proxy backend is the acceptance boundary.
- G-002: The existing UCX-direct EP workflow must remain stable for users who build and run EP without the proxy backend.
- G-003: The EP Python-facing workflow and module identity should remain stable; Phase 1 should not introduce a redesigned user entry point.
- G-004: Phase 1 targets one CPU proxy worker with N proxy channels, where N covers EP's logical device-side lanes.
- G-005: Proxy multi-worker scaling, channel-to-worker mapping, and UCX worker/QP selection are out of scope for Phase 1.
- G-006: Single-node multi-process validation is acceptable initial Phase 1 evidence if it demonstrates that the proxy backend path is configured and exercised.
- G-007: A real two-node proxy RDMA run is valuable follow-on evidence, but it is not required before initial Phase 1 completion.
- G-008: Elastic LL all-RDMA fallback under the proxy is acceptable for Phase 1; proxy-side `nixlGetPtr` and NVLink/P2P restoration are deferred.
- G-009: No minimum proxy performance threshold gates Phase 1, and no UCX-direct versus CPU-proxy performance artifact is required for Phase 1 acceptance.
- G-010: No new third-party dependency is expected for Phase 1; the work should use the existing NIXL GPU Device API proxy backend, CUDA device-linking support, UCX backend/provider, and EP test environment.
- G-011: Phase 1 should consume existing CPU proxy capabilities rather than changing CPU proxy internals; internal proxy scaling is a separate milestone.
- G-012: Later stages must define explicit evidence that elastic LL used the expected proxy all-RDMA fallback, because the exact signal is still unresolved.
- G-013: Later stages must define acceptable reduced-size EP configurations if default settings time out under the one-worker proxy model.

## Assumptions
- The staged plan in `ep-integ-plan.md` is source material, but Phase 1 acceptance is narrower than the full comparison and optimization plan.
- The candidate proxy channel count is N = `max(num_sms / 2, num_local_experts)`, matching the current understanding of HT and LL channel-id usage; architecture should verify this against the current EP code and any override behavior.
- The existing proxy path supports the device operations needed for HT and for elastic LL's all-RDMA fallback.
- Proxy `nixlGetPtr` does not currently restore the LL NVLink/P2P fast path, so Phase 1 treats that behavior as a known fallback rather than a defect.
- One CPU proxy worker may bottleneck EP workloads; if that blocks correctness validation, reduced test sizes or timeout adjustments may be acceptable only when they still prove the intended proxy path.
- Performance data remains useful for follow-on decisions, but it should not be used to reject Phase 1 if correctness is established.

## Phases
### Phase 1: Correctness-first EP-on-proxy
Enable the EP maintainer and NIXL CPU proxy maintainer to run the existing EP workflow through the CPU-proxy backend and validate HT plus elastic LL correctness. This phase keeps the UCX-direct workflow stable, uses one proxy worker with enough proxy channels for EP's logical lanes, starts with single-node multi-process validation, and accepts elastic LL's all-RDMA fallback.

### Phase 1.5: CPU proxy scaling follow-on
After Phase 1 correctness is green, evaluate CPU proxy internal scaling as a distinct follow-on milestone. This includes multiple proxy workers/channels, channel-to-worker mapping, UCX worker/QP selection, and transport thread-safety or progress behavior under multiple proxy threads.

### Phase 2: Optional LL peer-pointer restoration
Investigate whether proxy-side `nixlGetPtr` can provide device-usable peer pointers for NVLink/P2P-capable peers. Decide whether restoring the LL fast path is worth the complexity after Phase 1 behavior and later performance evidence are understood.
