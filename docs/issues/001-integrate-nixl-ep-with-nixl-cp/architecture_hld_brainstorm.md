---
issue-id: 001-integrate-nixl-ep-with-nixl-cp
stage: discuss_architecture_hld
timestamp: 2026-06-10T08:05:43Z
---

# Architecture HLD Brainstorm

## Decisions

1. Keep Phase 1 inside the existing EP rank process.
   - Components remain the existing `nixl_ep` Python/CUDA extension, EP CUDA kernels, NIXL agent, agent-owned CPU proxy runtime, UCX backend/provider, and manual validation harnesses.
   - No new daemon, external service, scheduler, control plane, runtime backend selector, or renamed Python module is introduced for Phase 1.

2. Use build-time backend selection.
   - The existing Meson GPU Device API backend selection remains the high-level switch between UCX-direct and CPU-proxy builds.
   - UCX-direct and proxy validation should use separate build trees or otherwise make the selected backend unambiguous.
   - The `nixl_ep` Python-facing workflow and module identity stay stable.

3. Keep EP kernels backend-agnostic.
   - HT and LL kernels continue to call the existing NIXL device wrappers.
   - The proxy build routes `nixlPut` and `nixlAtomicAdd` through the GPU-to-CPU proxy boundary as channelized work records.
   - Phase 1 does not fork HT/LL kernels for proxy-specific behavior.

4. Add proxy enablement to the existing EP host runtime lifecycle.
   - In proxy builds, EP initializes the existing NIXL agent with device proxy enabled.
   - The agent-owned CPU proxy runtime is started with one proxy worker and N proxy channels.
   - EP publishes the proxy device context after backend creation and clears it before agent teardown.
   - Proxy context publish/clear failures are setup failures, not deferred device-side surprises.

5. Preserve CPU proxy and UCX internals in Phase 1.
   - Phase 1 uses one proxy worker and N proxy channels, where N covers EP logical lanes.
   - UCX worker/QP scaling, channel-to-worker routing, and proxy multi-worker behavior are Phase 1.5 follow-ons.
   - The UCX backend/provider and CPU proxy runtime are consumed as existing capabilities unless implementation finds a correctness blocker.

6. Treat proxy channels as EP lane coverage, not a performance lane guarantee.
   - HT lane coverage is based on `num_sms / 2`.
   - LL lane coverage is based on local expert lanes.
   - The HLD expectation is `proxyChannelCount >= max(HT lanes, LL local expert lanes)`.
   - Operator overrides may exist, but an override below the derived minimum must fail before validation is accepted.

7. Reuse existing memview and metadata boundaries.
   - EP keeps its current local/remote memview preparation shape.
   - The NIXL agent/proxy path owns proxy memview ID indirection.
   - Validation should cover normal memview preparation and teardown, but Phase 1 should not redesign the memview model.

8. Accept an explicit HT-compatible proxy smoke as the initial Phase 1 HT evidence path.
   - A full two-node HT RDMA run remains valuable follow-on evidence.
   - The known true single-node fallback from the reference plan is not valid Phase 1 evidence under current HT constraints.
   - The accepted smoke must prove HT correctness and CPU-proxy activity together.

9. Require both EP-visible and proxy-runtime-visible LL fallback evidence.
   - EP-visible evidence proves the LL path used the accepted all-RDMA fallback rather than the deferred NVLink/P2P fast path.
   - Proxy-runtime-visible evidence proves CPU proxy activity occurred during the LL run.
   - LL correctness without both evidence classes is inconclusive, not accepted.

10. Classify invalid setup separately from inconclusive evidence.
   - Missing proxy setup, missing proxy context, unsupported topology, or under-provisioned channels should hard-fail early with an actionable reason.
   - Correctness pass without required proxy/fallback evidence should be classified as inconclusive.
   - Timeout under the one-worker proxy model is inconclusive unless the run uses pre-approved reduced-size criteria.

11. Add lightweight permanent correctness instrumentation now.
   - Evidence should be deterministic enough for maintainers to collect without log archaeology.
   - The intended scope is correctness instrumentation: backend selected, proxy worker/channel configuration, proxy worker activity, and LL fallback signal.
   - This is not performance telemetry, proxy throughput tuning, or a comparison artifact.

12. Preserve UCX-direct stability as an independent evidence path.
   - A small UCX-direct correctness smoke remains required after proxy integration.
   - UCX-direct stability is not proven by a later UCX-direct versus CPU-proxy performance comparison alone.

13. Defer peer-pointer restoration.
   - Proxy-side `nixlGetPtr` and NVLink/P2P restoration are Phase 2.
   - Any future peer-pointer design must define memory safety boundaries before exposing device-usable peer pointers.

14. Treat absent reference-plan paths as non-authoritative for implementation detail.
   - `ep-integ-plan.md` is useful staged source material, but several cited pingpong/docs paths are absent in this checkout.
   - HLD/LLD should prefer current in-repository CPU proxy, GPU Device API, EP, and test patterns over stale references.

## Open Questions

1. What exact HT-compatible proxy smoke should be implemented?
   - The HLD accepts a smoke as the initial Phase 1 evidence path, but LLD/tasks must specify whether it lives in the existing HT test path, a new EP-specific smoke, or a smaller validation wrapper.

2. What is the exact reduced-size correctness floor?
   - Reduced HT/LL runs may count only when the configuration is defined before validation and still proves the intended proxy path.
   - Tasks must name minimum rank, token, SM, and expert settings before relying on reduced-size evidence.

3. What exact instrumentation surface is least invasive?
   - The HLD requires lightweight permanent correctness evidence.
   - LLD must choose concrete locations for backend-selected evidence, proxy activity counting, channel coverage checks, and LL fallback signal.

4. How should Phase 2 peer-pointer safety be bounded?
   - If proxy `nixlGetPtr` later returns device-usable peer pointers, architecture must require rank/memview scoping and bounds enforcement before LLD changes contracts.

## Worked Shapes

### Preferred Phase 1 Runtime Shape

1. Maintainer configures EP for UCX-direct or CPU-proxy backend through the existing build option.
2. EP produces the same Python-facing `nixl_ep` workflow.
3. In proxy mode, each EP rank process creates the existing NIXL agent and UCX backend.
4. The agent starts the CPU proxy runtime with one worker and N channels.
5. EP publishes the proxy device context before device operations.
6. EP prepares local and remote memviews using the existing path.
7. HT and LL kernels issue backend-agnostic NIXL device operations.
8. CPU proxy resolves proxy memviews and submits UCX operations.
9. EP clears the proxy device context and releases memviews during teardown.

### HT Phase 1 Evidence Shape

1. Run the explicit HT-compatible proxy smoke accepted by LLD/tasks.
2. Prove HT correctness assertions pass.
3. Prove CPU-proxy activity occurred.
4. Reject the known invalid true single-node fallback.
5. Treat timeout-only outcomes as inconclusive unless a pre-approved reduced configuration was used.
6. Keep two-node HT RDMA validation as follow-on evidence, not the initial Phase 1 gate.

### Elastic LL Evidence Shape

1. Run elastic LL suite or accepted LL smoke against the proxy build.
2. Prove LL correctness passes.
3. Record backend selection in an EP-visible way.
4. Record all-RDMA fallback in an EP-visible way.
5. Record proxy worker activity in a proxy-runtime-visible way.
6. Treat a correctness pass without fallback or proxy activity evidence as inconclusive.

### Invalid Setup Shape

1. Detect missing proxy setup, missing proxy context, unsupported topology, missing fallback signal, missing proxy activity, or under-provisioned channels.
2. Hard-fail early for invalid setup that cannot prove the proxy path.
3. Mark passing-but-unproven runs as inconclusive.
4. Do not allow silent UCX-direct fallback to satisfy proxy evidence.

### Follow-On Shapes

1. Phase 1.5 may add multiple proxy workers, channel-to-worker mapping, UCX worker/QP selection, and multi-thread/progress validation.
2. Phase 2 may explore proxy-side `nixlGetPtr` only after correctness is proven and peer-pointer safety boundaries are explicit.
3. Performance comparison remains useful but separate from Phase 1 correctness acceptance.

## Panel

### product

- Emphasized named actors: EP maintainer and NIXL CPU proxy maintainer.
- Pushed for maintainer-visible evidence rather than correctness-only green status.
- Flagged reduced-size criteria and evidence classification as product-visible trust risks.
- Confidence: low before human convergence because acceptance evidence was still under-specified.

### system-arch

- Recommended no new service or control plane.
- Placed Phase 1 inside existing EP rank processes with one agent-managed CPU proxy runtime per rank.
- Separated Phase 1 from multi-worker proxy scaling and Phase 2 peer-pointer restoration.
- Confidence: low before human convergence because HT and LL evidence boundaries needed explicit choices.

### sw-arch

- Recommended reuse/adapt of existing EP build target, EP host lifecycle, device wrappers, memview shape, CPU proxy runtime, and UCX proxy adapter.
- Identified new validation evidence surfaces as the main needed addition.
- Flagged missing proxy context and under-provisioned channels as setup failures that should not surface as late device assertions.
- Confidence: low before human convergence because smoke placement and fallback evidence seams were unresolved.

### sw-dev

- Rated Phase 1 build and lifecycle work as medium cost, with higher validation risk around HT topology and evidence.
- Recommended deterministic evidence over manual log archaeology.
- Flagged dual-build import confusion, brittle manual proxy evidence, and one-worker timeout handling as developer workflow risks.
- Confidence: low before human convergence because LLD needs pinned HT and LL acceptance-proof seams.

## Human Convergence

The human accepted the following convergence choices during Beat 2.5:

1. Phase 1 should accept a new explicit HT-compatible proxy smoke first; two-node HT RDMA remains follow-on evidence.
2. LL fallback evidence should be both EP-visible and proxy-runtime-visible.
3. Invalid setup should hard-fail; passing runs missing required evidence should be inconclusive.
4. Phase 1 should add lightweight permanent correctness instrumentation now.
