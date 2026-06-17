# Pipeline Q&A Answers

**Pipeline:** 001-integrate-nixl-ep-with-nixl-cp
**Generated:** 2026-06-10 11:16 UTC
**Issue type:** feature
**Topology:** topology_new_feature_detailed__interactive
**Cost profile:** high

## Brainstorm

- **Feature description:** Integrate the NIXL EP example with the NIXL CPU Proxy GPU device backend for a correctness-first Phase 1. The initial target is EP running through CPU proxy with 1 proxy worker and N proxy channels, where N covers EP's logical lanes; performance scaling is deferred.
- **Impacted codebase:** Phase 1 should mainly impact examples/device/ep/meson.build, examples/device/ep/csrc/nixl_ep.cpp, and a new proxy context publish wrapper under examples/device/ep/csrc/kernels/. It should consume existing proxy APIs in src/api/gpu/proxy/, src/core/device_proxy/, and src/plugins/ucx/device_proxy/ without changing CPU proxy internals. Phase 1.5 can then address internal CPU proxy infrastructure for multiple workers/channels and channel-to-backend-worker mapping.
- **Dependencies / integrations:** No new third-party dependency is expected for Phase 1. The integration depends on existing CUDA device linking, the NIXL GPU Device API proxy backend, the UCX backend/provider, and EP's CUDA/UCX test environment. Validation should use separate UCX-direct and proxy builds, with correctness as the Phase 1 gate.
- **References / contracts:** Reference ep-integ-plan.md, but treat it as a staged plan: Phase 1 is correctness-only EP-on-proxy with proxyWorkerCount=1 and proxyChannelCount=N; Phase 1.5 is CPU proxy infrastructure improvement for multiple workers/channels; Phase 2 is optional nixlGetPtr/NVLink P2P restoration after correctness and baseline behavior are understood.

## Setup (Governance, Topology, Cost Profile)

- **Cost profile:** High — maximum quality; strongest model/review budget
- **Documentation set:** High-docs — full docs; architecture split into HLD + LLD
- **Doc-set customization:** Looks good — use suggested set
- **Governance approach:** Interactive [experimental] — we discuss each artifact together (mid-panel must_ask elicitation) before proceeding
- **Reviewer strategy:** Max — full panel; most thorough
- **Run UI mode:** With UI — open/reuse the local Run UI for progress and debug visibility
- **SUBAGENT_APPROVAL:** Yes - allow relay-local sub-agents for pipeline jobs
- **Verbosity level:** High-verb — full coverage; context included
