# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for EP CPU-proxy Phase 1 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

EVIDENCE_KIND = "ep_proxy_evidence_v1"
BACKEND_SOURCE = "loaded_extension_compile_time_backend"
BACKEND_API = "get_gpu_device_api_backend"

HT_PROXY_PATHS = {"ht_proxy_smoke", "ht_two_node_rdma"}
PROXY_PATHS = HT_PROXY_PATHS | {"elastic_ll"}
VALIDATION_PATHS = PROXY_PATHS | {"ucx_direct_smoke"}


def proxy_activity_delta(before: int | None, after: int | None) -> int:
    if before is None or after is None:
        return 0
    return max(0, int(after) - int(before))


def make_ep_proxy_evidence(
    *,
    backend: str,
    rank: int,
    validation_path: str,
    correctness: str,
    proxy_activity_before: int | None = None,
    proxy_activity_after: int | None = None,
    proxy_activity_submitted_work_count: int | None = None,
    proxy_context_published: bool | None = None,
    proxy_context_owner_id: int | None = None,
    proxy_worker_count: int | None = None,
    proxy_channel_count: int | None = None,
    backend_source: str = BACKEND_SOURCE,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if proxy_activity_submitted_work_count is None:
        proxy_activity_submitted_work_count = proxy_activity_delta(
            proxy_activity_before, proxy_activity_after
        )

    record: dict[str, Any] = {
        "kind": EVIDENCE_KIND,
        "backend": backend,
        "backend_source": backend_source,
        "loaded_extension_backend_api": BACKEND_API,
        "rank": rank,
        "validation_path": validation_path,
        "correctness": correctness,
    }

    if validation_path in PROXY_PATHS:
        record.update(
            {
                "proxy_worker_count": proxy_worker_count,
                "proxy_channel_count": proxy_channel_count,
                "proxy_context_owner_id": proxy_context_owner_id,
                "proxy_context_published": proxy_context_published,
                "proxy_activity_observed": (proxy_activity_submitted_work_count > 0),
                "proxy_activity_source": "proxy_worker_submission_counter",
                "proxy_activity_submitted_work_count": (
                    proxy_activity_submitted_work_count
                ),
                "proxy_scheduler": "bounded_one_submission_per_channel_per_cycle",
            }
        )
    else:
        record.update(
            {
                "proxy_activity_observed": False,
                "proxy_activity_source": "not_applicable",
                "proxy_activity_submitted_work_count": 0,
            }
        )

    if extra:
        record.update(dict(extra))

    return classify_ep_proxy_evidence(record)


def classify_ep_proxy_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    classification, reason = _classify_ep_proxy_evidence(out)
    out["classification"] = classification
    out["reason"] = reason
    return out


def _classify_ep_proxy_evidence(record: Mapping[str, Any]) -> tuple[str, str]:
    blocked_reason = record.get("blocked_reason")
    if blocked_reason:
        return "blocked", str(blocked_reason)

    setup_error = record.get("setup_error")
    if setup_error:
        return "failed", str(setup_error)

    correctness = record.get("correctness")
    if correctness == "fail":
        return "failed", "correctness check failed"
    if correctness == "not_run":
        return "blocked", "correctness check did not run"
    if correctness != "pass":
        return "inconclusive", "correctness result is missing or unknown"

    validation_path = record.get("validation_path")
    if validation_path not in VALIDATION_PATHS:
        return "blocked", "validation path is not an accepted Phase 1 path"

    if record.get("backend_source") != BACKEND_SOURCE:
        return (
            "inconclusive",
            "backend was not sourced from the loaded extension getter",
        )

    backend = record.get("backend")
    if backend not in ("proxy", "ucx"):
        return "inconclusive", "loaded extension backend is missing or unknown"

    if validation_path == "ucx_direct_smoke":
        if backend != "ucx":
            return "failed", "UCX-direct smoke did not load the UCX backend"
        return "accepted", "UCX-direct correctness smoke passed"

    if record.get("unsupported_single_node_fallback"):
        return (
            "inconclusive",
            "known invalid single-node HT fallback is not accepted evidence",
        )

    if backend != "proxy":
        return "failed", "proxy validation did not load the proxy backend"

    if record.get("proxy_context_published") is not True:
        return "failed", "proxy device context was not published"

    if record.get("proxy_activity_observed") is not True:
        return (
            "inconclusive",
            "correctness passed but proxy worker activity was not observed",
        )

    if validation_path == "elastic_ll":
        return (
            "accepted",
            "elastic LL correctness passed with proxy activity evidence",
        )

    return "accepted", "HT correctness passed with proxy activity evidence"


def evidence_output_path(
    output: str | Path,
    *,
    rank: int | None = None,
    phase: int | str | None = None,
) -> Path:
    path = Path(output)
    if rank is None and phase is None:
        return path
    if not path.suffix:
        name_parts = ["ep_proxy_evidence"]
        if rank is not None:
            name_parts.append(f"rank{rank}")
        if phase is not None:
            name_parts.append(f"phase{phase}")
        return path / (".".join(name_parts) + ".json")

    suffix_parts = []
    if rank is not None:
        suffix_parts.append(f"rank{rank}")
    if phase is not None:
        suffix_parts.append(f"phase{phase}")
    return path.with_name(f"{path.stem}.{'.'.join(suffix_parts)}{path.suffix}")


def write_evidence_record(
    record: Mapping[str, Any],
    output: str | Path,
    *,
    rank: int | None = None,
    phase: int | str | None = None,
) -> Path:
    path = evidence_output_path(output, rank=rank, phase=phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(record), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
