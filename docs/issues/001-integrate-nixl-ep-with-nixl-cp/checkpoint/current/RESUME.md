# Resume Pipeline `001-integrate-nixl-ep-with-nixl-cp`

- **Skill:** `wf-feature`
- **Stage:** `done`
- **Updated at:** 2026-06-16T13:10:00Z
- **Updated by:** tdavidor@adv-dev-420

## Resume command

From any host (Claude Code / Codex / Cursor / OpenCode), ask the agent:

    with nv-sdd, resume pipeline in folder 001-integrate-nixl-ep-with-nixl-cp

The nv-sdd router will present a calibration-override prompt
and dispatch to the correct skill with:

    continue pipeline 001-integrate-nixl-ep-with-nixl-cp

If the runtime `.tmp/` state is missing but this slot exists, restore it with:

    engine-restore --pipeline-id 001-integrate-nixl-ep-with-nixl-cp --checkpoint-id current

## What this slot holds

- ``manifest.yaml`` — pipeline id, skill, stage, timestamps (router dispatch)
- ``e2r/`` — latest active E2R node folder
- ``pipeline_state.json`` — FSM cursor + pipeline data

Older snapshots from this pipeline rotate into
``.tmp/001-integrate-nixl-ep-with-nixl-cp/checkpoint-history/`` (gitignored, ring buffer).
