---
issue-id: 001-integrate-nixl-ep-with-nixl-cp
stage: discuss_tasks
timestamp: 2026-06-10T11:08:52Z
human-dialog-mode: current_conversation
---

# Tasks Brainstorm

## Outcome

The panel converged on a correctness-first Phase 1 task split. Phase 1 should
make the existing EP workflow runnable through the NIXL CPU-proxy GPU Device API
backend, prove HT and elastic LL correctness with deterministic proxy evidence,
and keep UCX-direct correctness green as a separate smoke.

The human decision for the remaining HT ambiguity is:

- Phase 1 accepts a dedicated single-node/multiprocess HT-compatible proxy smoke
  as the HT correctness artifact.
- A real two-node RDMA HT run is deferred to a later validation or performance
  artifact and is not required for initial Phase 1 acceptance.

This means the HT task must add an explicit compatible smoke path. The known
true single-node HT fallback remains rejected or inconclusive unless the new
smoke path satisfies the current HT constraints and records the required proxy
evidence.

## Source Inputs

- `goal.md`: Phase 1 is correctness-first; performance, multi-worker proxy
  scaling, and two-node comparison artifacts are follow-ons.
- `use-cases.md`: UC-1 through UC-5 define selected-backend builds, HT proxy
  validation, elastic LL all-RDMA fallback validation, UCX-direct preservation,
  and invalid setup classification.
- `architecture-hld.md`: Phase 1 stays inside each EP rank process, uses
  build-time backend selection, one proxy worker with N proxy channels, and a
  validation evidence boundary.
- `architecture-lld.md`: Defines the concrete seams for backend introspection,
  owner-aware proxy publish/clear, explicit proxy lane ceiling, proxy lifecycle,
  validation-only proxy activity, LL fallback evidence, and
  `ep_proxy_evidence_v1`.

## Settled Decisions

- First user-visible progress should be a proxy selected-backend build/import
  slice: the NIXL CPU proxy maintainer can build the existing `nixl_ep` workflow
  in proxy mode and the loaded extension reports backend `proxy`.
- Final Phase 1 acceptance is not build success. It requires HT proxy
  correctness, elastic LL proxy all-RDMA fallback correctness, and independent
  UCX-direct correctness evidence.
- Backend selection remains build-time. Phase 1 does not introduce a renamed
  Python module, a redesigned user entry point, a runtime backend toggle, or a
  new daemon.
- Phase 1 uses one CPU proxy worker with N proxy channels. N must cover EP's
  logical device-side lanes. Multiple workers, channel-to-worker mapping, UCX
  worker/QP routing, and proxy performance tuning are Phase 1.5 follow-ons.
- `NIXL_EP_PROXY_CHANNELS` is a proxy-only override. `NIXL_EP_NUM_CHANNELS`
  remains UCX-direct device-channel configuration and must not be reused as the
  proxy channel override.
- `proxy_lane_ceiling` should be explicit and separate from
  `num_experts_per_rank`; HT passes its lane requirement and elastic LL passes
  its local expert lane requirement.
- Correctness without evidence is not accepted. HT needs loaded-extension proxy
  backend selection, context publish, channel coverage, proxy activity, and
  correctness. Elastic LL also needs explicit all-RDMA fallback evidence.
- `ep_proxy_evidence_v1` is a validation/test artifact, not a production public
  API. Default placement should be under the EP validation/test surface unless
  implementation proves a package-private helper is cleaner.
- The deterministic proxy activity counter is the only Phase 1 proxy-internal
  scope exception. It should be the smallest reset/snapshot surface needed to
  prove supported proxy submissions happened during the validated run.
- Reduced-size validation cannot be accepted ad hoc. The dedicated HT smoke
  defines the approved Phase 1 HT correctness floor; other timeout workarounds
  remain inconclusive unless approved before the run.
- Evidence completeness should be reviewed by both the EP maintainer and the
  NIXL CPU proxy maintainer because acceptance crosses EP behavior and proxy
  runtime evidence.

## Recommended Task List

### T1: Selected-backend EP build and loaded-backend introspection

**Trace:** UC-1, HLD Backend Build Boundary, LLD EP Backend Build Wiring.

**Scope:** Allow EP to build for `ucx` and `proxy`, preserve the `none` skip,
include/link proxy-only source and device library only in proxy builds, and
expose `nixl_ep.get_gpu_device_api_backend()` from the loaded extension.

**TDD anchor:** Proxy and UCX build/import tests fail until the same `nixl_ep`
module imports and reports `proxy` or `ucx` from the loaded extension.

**Done:** Proxy build imports through the existing workflow and reports `proxy`;
UCX-direct build imports and reports `ucx`; UCX-direct build has no proxy
publish/link leakage.

**Risk:** Medium. Meson/backend selection and CUDA device linking are sensitive.

### T2: Owner-aware proxy publish/clear CUDA seam

**Trace:** LLD Proxy Context Publish/Clear CUDA Seam.

**Scope:** Add host-callable CUDA wrappers:
`nixl_ep_proxy_publish_context(void *ctx, uint64_t owner_id)` and
`nixl_ep_proxy_clear_context(uint64_t owner_id)`. Enforce one active proxy
`Buffer` owner per CUDA context/rank process.

**TDD anchor:** Proxy device API tests fail until null context, second-owner
publish, wrong-owner clear, owner clear, and republish behavior match the LLD.

**Done:** Proxy context setup fails early on invalid ownership; clear happens
before proxy teardown and cannot clear another owner's context.

**Risk:** High. The global CUDA symbol and teardown ordering are correctness
critical.

### T3: Explicit proxy lane ceiling and proxy channel validation

**Trace:** UC-1, UC-5, LLD Proxy Channel Sizing and Ordering Contract.

**Scope:** Add `proxy_lane_ceiling` through Python/C++
`Buffer.update_memory_buffers(...)`; store it separately from
`num_experts_per_rank`; derive `required_proxy_channels` from it; validate
`NIXL_EP_PROXY_CHANNELS` only for proxy builds.

**TDD anchor:** Binding/config tests fail until missing or zero lane ceiling,
invalid proxy-channel override, and under-provisioned override fail in proxy
mode before kernels run.

**Done:** Required and configured proxy channel counts are evidence-visible;
UCX-direct channel configuration remains independent.

**Risk:** High. This touches the Python API, C++ storage, and validation caller
migration.

### T4: EP host proxy lifecycle in `Buffer`

**Trace:** UC-1, UC-2, UC-3, HLD NIXL Agent and Backend Runtime, LLD EP Host
Runtime Proxy Lifecycle.

**Scope:** In proxy builds, enable device proxy in the agent, configure one
proxy worker with N channels, create the UCX backend with `num_workers=1`,
publish the proxy device context after backend creation, and clear it before
teardown. UCX-direct builds must avoid proxy setup.

**TDD anchor:** Proxy `Buffer` init smoke fails until evidence reports worker
count 1, sufficient channel count, context published, and no silent UCX-direct
fallback.

**Done:** Proxy runtime lifecycle is owned by the EP rank process; UCX-direct
workflow remains stable.

**Risk:** High. This crosses agent setup, backend creation, proxy context,
memview setup, and teardown.

### T5: Update HT and elastic LL callers for lane ceiling

**Trace:** UC-2, UC-3, LLD Proxy Channel Sizing and Ordering Contract.

**Scope:** Update HT validation to pass its proxy lane requirement, normally
`num_qps_per_rank`. Update elastic LL validation to pass its local expert lane
requirement. Shared paths should pass the maximum requirement when both are
active.

**TDD anchor:** HT/elastic setup tests fail until proxy lane ceiling is explicit
and sufficient for each validation path.

**Done:** No proxy validation path depends on implicit expert metadata for
proxy channel sizing.

**Risk:** Medium. Caller migration must avoid changing UCX-direct behavior.

### T6: Deterministic proxy activity counter

**Trace:** UC-2, UC-3, UC-5, LLD Proxy Runtime and UCX Provider Boundary.

**Scope:** Add the smallest validation-visible reset/snapshot counter at the
proxy backend submission boundary. Increment when the proxy worker submits
supported PUT or ATOMIC_ADD work to the backend. Do not change worker scheduling
or UCX worker/QP routing.

**TDD anchor:** Proxy runtime tests fail until PUT and ATOMIC_ADD submissions
increment the counter, while runtime creation alone does not.

**Done:** Logs and runtime startup messages are not needed to prove activity;
current drain-until-empty scheduling semantics remain unchanged.

**Risk:** Medium. The counter must prove proxy execution without becoming a new
production control surface.

### T7: LL all-RDMA fallback signal

**Trace:** UC-3, LLD EP Memview and Device Operation Contract, LLD Validation
Evidence Surface.

**Scope:** Record an explicit LL fallback branch signal when proxy
`nixlGetPtr` cannot provide a device-usable peer pointer and LL selects the
accepted all-RDMA fallback.

**TDD anchor:** LL evidence tests fail until proxy fallback is observed and
P2P/UCX-direct paths do not falsely count it.

**Done:** Elastic LL correctness without fallback evidence is classified
inconclusive, not accepted.

**Risk:** High. Instrumentation touches LL execution path selection and must not
turn into Phase 1 peer-pointer restoration.

### T8: `ep_proxy_evidence_v1` record and classifier

**Trace:** UC-2 through UC-5, HLD Validation and Evidence Boundary, LLD
Validation Evidence Surface.

**Scope:** Emit and classify `ep_proxy_evidence_v1` from loaded backend,
context ownership, channel sizing, proxy activity, LL fallback, correctness,
validation path metadata, and failure reasons.

**TDD anchor:** Classifier fixtures fail until HT pass without proxy activity is
inconclusive, LL pass without fallback is inconclusive, correctness failure is
failed, invalid setup is failed or blocked, and UCX smoke pass is accepted for
UCX-direct only.

**Done:** Every accepted, failed, blocked, or inconclusive result has an
actionable reason. Out-of-band backend evidence is not accepted.

**Risk:** Medium. This becomes the maintainer-facing acceptance surface.

### T9: Dedicated HT-compatible proxy smoke

**Trace:** UC-2, Goal G-006/G-007, LLD EP Validation Harnesses.

**Scope:** Add a single-node/multiprocess HT-compatible proxy smoke under
`examples/device/ep/tests/` that satisfies current HT constraints and uses the
proxy backend path. Reject the known invalid true single-node fallback and mark
unsupported topology or missing evidence as inconclusive/blocked.

**TDD anchor:** Validation-path tests fail until unsupported single-node runs
are rejected or inconclusive, and the new smoke produces HT correctness plus
proxy evidence.

**Done:** The EP maintainer can produce accepted HT Phase 1 evidence without a
real two-node RDMA setup. Real two-node RDMA remains a later validation
artifact.

**Risk:** High. The smoke must be small enough to run reliably with one proxy
worker while still proving the intended proxy path.

### T10: Elastic LL proxy validation path

**Trace:** UC-3, LLD EP Validation Harnesses.

**Scope:** Wire the accepted elastic LL suite or smoke to collect loaded
backend, proxy activity, all-RDMA fallback, and correctness evidence.

**TDD anchor:** Elastic validation fails until proxy run records activity and
fallback evidence; missing either classifies as inconclusive.

**Done:** Elastic LL all-RDMA fallback is accepted only when explicit evidence
is present.

**Risk:** High. The validation path must distinguish accepted fallback from
accidental success or UCX-direct execution.

### T11: Independent UCX-direct correctness smoke

**Trace:** UC-4, HLD Preserve UCX-direct correctness separately.

**Scope:** Keep a small UCX-direct correctness smoke separate from proxy
evidence and separate from any later UCX-direct versus proxy performance
artifact.

**TDD anchor:** UCX build/import plus selected smoke fails until backend is
`ucx` and correctness passes through the unchanged workflow.

**Done:** UCX-direct regression blocks Phase 1 acceptance.

**Risk:** Low. The main risk is accidentally mixing UCX-direct evidence into
proxy evidence.

### T12: Invalid setup and inconclusive-evidence tests

**Trace:** UC-5, LLD EP Validation Harnesses.

**Scope:** Cover missing lane ceiling, invalid/under-provisioned proxy channels,
missing context, overlapping proxy `Buffer` owners, missing proxy activity,
missing LL fallback evidence, out-of-band backend evidence, unsupported HT
topology, and unapproved reduced-size runs.

**TDD anchor:** Negative tests fail until invalid setup fails early and
correctness-without-evidence is classified inconclusive.

**Done:** Maintainers cannot accidentally accept silent UCX-direct fallback,
missing proxy activity, unsupported topology, or timeout workarounds as Phase 1
proxy evidence.

**Risk:** Medium. Negative tests are the guardrail against false acceptance.

## Suggested Review Slices

1. **Build signal:** T1 only. This gives the first observable proxy build/import
   signal and unblocks proxy-only linkage.
2. **Runtime ownership:** T2, then T3, then T4. These share the `Buffer` and
   publish/lifecycle surface and should be serialized.
3. **Caller migration:** T5. Keep this close to T3 so no proxy validation path
   can run with implicit channel sizing.
4. **Evidence foundations:** T6 and T7, then T8. T6 mostly lives in
   `src/core/device_proxy/*`; T7/T8 touch EP validation surfaces. Do not mix T6
   with multi-worker scheduling changes.
5. **Validation harnesses:** T9, T10, T11, and T12 after the evidence record is
   available.

## Shared Edit Zones

- `examples/device/ep/meson.build`: primarily T1.
- `examples/device/ep/csrc/nixl_ep.cpp`,
  `examples/device/ep/csrc/nixl_ep.hpp`, and
  `examples/device/ep/nixl_ep/buffer.py`: T2 through T5 and parts of T8; these
  should be serialized.
- `src/core/device_proxy/*`: T6 activity counter only; avoid worker scheduling,
  multi-worker, or UCX routing changes in Phase 1.
- `examples/device/ep/tests/test_ht.py`,
  `examples/device/ep/tests/elastic/elastic.py`, and shared evidence helpers:
  T8 through T12; serialize evidence helper changes before HT/LL/UCX harness
  work.

## Phase 1 Acceptance

Phase 1 can be accepted only when all of these are true:

- Proxy selected-backend build imports as `nixl_ep` and the loaded extension
  reports backend `proxy`.
- UCX-direct selected-backend build imports as `nixl_ep` and the loaded
  extension reports backend `ucx`.
- Proxy `Buffer` lifecycle publishes and clears the owner-bound proxy device
  context.
- Proxy validation uses one worker and enough proxy channels for the explicit
  `proxy_lane_ceiling`.
- Dedicated HT proxy smoke passes correctness and records accepted
  `ep_proxy_evidence_v1`.
- Elastic LL proxy validation passes correctness and records proxy activity plus
  explicit all-RDMA fallback evidence.
- UCX-direct correctness smoke remains green independently.
- Invalid setup and missing evidence classify as failed, blocked, or
  inconclusive instead of accepted.

## Deferred Work

- Real two-node RDMA HT validation and any two-node comparison artifact.
- UCX-direct versus CPU-proxy performance artifact.
- CPU proxy multi-worker scaling, channel-to-worker mapping, UCX worker/QP
  routing, and related thread-safety/progress work.
- Restoring proxy-side `nixlGetPtr` peer-pointer or NVLink/P2P LL fast path.
- Fair round-robin or bounded proxy channel polling. Current Phase 1 accepts
  completed correctness runs with evidence; skew-shaped timeouts remain
  inconclusive unless a later task changes scheduling or approves a reduced
  validation floor.

## Panel Notes

- Product seat emphasized the first observable build/import signal, named
  maintainers, and avoiding performance or two-node requirements in Phase 1.
- System-architecture seat emphasized that HT validation is the only external
  validation-path dependency and should be treated as a task boundary, not a
  deployment dependency.
- Software-architecture seat emphasized the ordering of new seams before their
  consumers: backend introspection, publish/clear, lane ceiling, lifecycle,
  activity/fallback signals, then evidence classifier.
- Software-development seat emphasized TDD anchors, teardown/ownership risk,
  channel-sizing compatibility risk, and the need for negative tests that block
  false acceptance.

## Human Convergence

One question was asked during task convergence:

- **Question:** For Phase 1 HT validation, should the task plan accept a
  dedicated single-node/multiprocess HT-compatible proxy smoke as the
  correctness artifact, with real two-node RDMA validation deferred, or should
  two-node RDMA be required for Phase 1 acceptance?
- **Answer:** Accept the dedicated smoke for Phase 1; defer real two-node RDMA
  validation.

## Remaining Implementation-Time Defaults

These are not blockers for task planning, but should be treated as defaults in
implementation unless local code constraints prove otherwise:

- Put `ep_proxy_evidence_v1` helpers under the EP validation/test surface rather
  than making them a production public API.
- Surface proxy activity through the smallest proxy runtime reset/snapshot seam
  that can be consumed by EP validation.
- Keep `proxy_lane_ceiling` optional for UCX-direct compatibility but mandatory
  for proxy validation once the loaded backend is proxy.
- Use the surrounding EP host-runtime error style for setup failures, but make
  all invalid setup and inconclusive evidence visible in the validation
  classifier.
