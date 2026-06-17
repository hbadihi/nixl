# Protocol V2 Operator Instructions

## Purpose

Complete the operator-owned response for an active `error` protocol node.

## Instructions

Node context is available in `/scrap/cpu-proxy/nixl/.tmp/001-integrate-nixl-ep-with-nixl-cp/relay/129_E2R_done/operator-context.json`.
The response contract is available in `/scrap/cpu-proxy/nixl/.tmp/001-integrate-nixl-ep-with-nixl-cp/relay/129_E2R_done/response-contract.json`.
Write the next R2E to `/scrap/cpu-proxy/nixl/.tmp/001-integrate-nixl-ep-with-nixl-cp/relay/130_R2E_recovery_ack/protocol.json` using only a top-level `control`
object.

Required control fields:

- `control.response_kind`: one of recovery_ack.
- `control.responds_to_file`: `/scrap/cpu-proxy/nixl/.tmp/001-integrate-nixl-ep-with-nixl-cp/relay/129_E2R_done/protocol.json`.
- `control.relay_status`: a terminal relay status accepted by the response
  contract.

For recovery acknowledgement, set `control.response_kind` to `recovery_ack`.
Set `control.relay_status` to `completed` after applying the recovery decision.
Set `control.decision` to the selected gate option or recovery action when
`control.relay_status` is `completed`.
For recoverable missing-output, missing-evidence, stale-output, or malformed-R2E
conditions, create or fix the required files and retry.
Use `control.relay_status: cancelled` only when the human explicitly asks to
stop/cancel or an unrecoverable external condition prevents continuation. Include
`control.reason` when cancelling.


After writing the R2E file, advance the engine with that exact file path.

## Boundaries

Do not pass inline JSON or legacy top-level `answers`, `decision`,
`job_status`, `job_outputs`, or `cancelled` keys as the response payload.
