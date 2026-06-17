---
issue-id: 001-integrate-nixl-ep-with-nixl-cp
stage: discuss_use_cases
timestamp: 2026-06-09T16:15:22Z
---

# Use Cases Brainstorm: Integrate NIXL EP With NIXL CPU Proxy

## Decisions

1. Phase 1 use cases are manual maintainer validation flows.

   The initiating actors are the EP maintainer and NIXL CPU proxy maintainer. CI or scheduler-triggered validation is not a Phase 1 actor unless a later stage explicitly promotes it.

2. UC-1: Build EP for UCX-direct or CPU-proxy backend.

   - Actor: NIXL CPU proxy maintainer.
   - Trigger: The maintainer configures/builds EP using the existing backend-selection workflow for either UCX-direct or CPU-proxy mode.
   - Outcome: The proxy build is runnable, the UCX-direct build remains stable, and the Python-facing `nixl_ep` workflow/module identity is unchanged.
   - Success signals carried: proxy backend configured; UCX-direct workflow stability; stable EP user entry point.

3. UC-2: Validate HT correctness through CPU proxy.

   - Actor: EP maintainer or NIXL CPU proxy maintainer.
   - Trigger: The maintainer runs HT against the proxy build using an explicitly accepted HT-compatible proxy smoke path.
   - Outcome: HT correctness passes, and evidence shows the CPU-proxy backend path was exercised.
   - Phase 1 note: A full two-node RDMA HT run is valuable follow-on evidence, but an explicit HT-compatible proxy smoke is sufficient for initial Phase 1 acceptance because multi-node setup is more complex.
   - Success signals carried: HT correctness; proxy path exercised.

4. UC-3: Validate elastic LL correctness through CPU-proxy all-RDMA fallback.

   - Actor: EP maintainer or NIXL CPU proxy maintainer.
   - Trigger: The maintainer runs the elastic LL suite, or an accepted elastic LL smoke, against the proxy build.
   - Outcome: Elastic LL correctness passes and validation captures evidence that the accepted all-RDMA proxy fallback was used.
   - Required evidence shape: proxy backend selected, proxy worker activity observed during LL, and an explicit fallback branch log/counter showing the all-RDMA proxy fallback was used.
   - Success signals carried: elastic LL correctness; accepted fallback path proven.

5. UC-4: Preserve UCX-direct correctness.

   - Actor: EP maintainer.
   - Trigger: The maintainer runs a small UCX-direct smoke/regression after proxy integration work.
   - Outcome: UCX-direct correctness remains green independently of any deferred UCX-direct versus CPU-proxy performance comparison.
   - Success signals carried: existing UCX-direct workflow stability.

6. UC-5: Fail clearly on invalid proxy validation setup.

   - Actor: EP maintainer or NIXL CPU proxy maintainer.
   - Trigger: The maintainer runs a proxy validation path with insufficient channels, missing proxy context setup, unsupported topology, absent fallback evidence, or another invalid Phase 1 setup.
   - Outcome: The run fails or is classified as inconclusive with an actionable reason; it must not silently pass, fall back to UCX-direct, skip the proxy path, or hang indefinitely.
   - Success signals carried: proxy validation evidence is trustworthy rather than accidental.

7. Reduced-size timeout criteria remain open.

   The user skipped defining minimum reduced-size configurations at this stage. Later architecture/tasks must decide whether and when reduced-size runs count as meaningful correctness evidence if the one-worker proxy path times out on default EP configurations.

## Open Questions

1. Which minimum reduced-size EP configurations count as valid correctness evidence if one-worker proxy mode times out at default settings?

   This was intentionally left unresolved during use-case elicitation. It must be settled before validation tasks treat reduced-size runs as Phase 1 evidence.

2. What exact implementation surface should produce the LL fallback log/counter?

   The use-case artifact defines the required actor-visible evidence shape but leaves the concrete mechanism to architecture/tasks.

3. What exact HT-compatible proxy smoke should satisfy UC-2?

   The user accepted a smoke path for Phase 1, but later stages must specify whether that is an existing test mode, a new small smoke, or a constrained run of the existing HT workflow.

## Worked Shapes

### Happy Path: HT Proxy Smoke

1. Maintainer builds EP with the CPU-proxy backend selected.
2. Maintainer runs the accepted HT-compatible proxy smoke.
3. The run exits successfully.
4. HT correctness assertions pass.
5. Evidence shows CPU-proxy backend path activity.
6. The run counts as Phase 1 HT correctness evidence.

### Happy Path: Elastic LL Proxy Fallback

1. Maintainer builds EP with the CPU-proxy backend selected.
2. Maintainer runs the accepted elastic LL suite or smoke.
3. The run exits successfully.
4. Elastic LL correctness assertions pass.
5. Validation records proxy worker activity during LL.
6. Validation records an explicit all-RDMA fallback branch log/counter.
7. The run counts as Phase 1 elastic LL correctness evidence.

### Happy Path: UCX-Direct Smoke

1. Maintainer builds EP in UCX-direct mode.
2. Maintainer runs a small UCX-direct smoke/regression.
3. The run exits successfully with unchanged Python-facing EP workflow/module identity.
4. The run counts as UCX-direct stability evidence without requiring a performance comparison artifact.

### Exception: Missing Proxy Evidence

If HT or elastic LL correctness passes but proxy-path evidence is missing, the result is inconclusive rather than accepted. Phase 1 evidence must show that the intended proxy path was exercised.

### Exception: Timeout Under One Worker

If a proxy run times out under the one-worker model, the result is validation-blocked or inconclusive unless the run uses a pre-approved reduced configuration. Because reduced-size criteria are not yet defined, later stages must define them before relying on such runs.

## Panel

- product: Proposed build, HT proxy, elastic LL fallback, and UCX-direct stability use cases with goal-to-UC traceability.
- system-arch: Confirmed Phase 1 flows are manual maintainer actions and no CI/scheduler actor is included for v1.
- sw-arch: Added exception semantics for missing path evidence, topology ambiguity, and timeout classification.
- sw-dev: Added explicit invalid-setup failure use case and emphasized that correctness without proxy/fallback evidence is not acceptable.

## Elicitation Summary

- HT Phase 1 evidence may use an explicit HT-compatible proxy smoke; two-node RDMA is follow-on.
- LL fallback evidence should combine proxy backend selection, proxy worker activity, and an explicit all-RDMA fallback branch log/counter.
- Reduced-size timeout criteria were skipped and remain open for later architecture/tasks.
- UCX-direct stability should be a small smoke/regression, separate from any UCX-vs-proxy comparison.
- Phase 1 use cases are manual maintainer actions only, not CI/scheduler-triggered flows.
