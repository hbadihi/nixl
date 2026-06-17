# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path

from proxy_evidence import (
    BACKEND_SOURCE,
    classify_ep_proxy_evidence,
    make_ep_proxy_evidence,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


class ProxyEvidenceClassifierTest(unittest.TestCase):
    def test_ht_pass_without_activity_is_inconclusive(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="ht_proxy_smoke",
            correctness="pass",
            proxy_activity_submitted_work_count=0,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=12,
            required_proxy_channels=12,
        )

        self.assertEqual(record["classification"], "inconclusive")
        self.assertIn("activity", record["reason"])

    def test_ht_pass_with_activity_is_accepted(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="ht_proxy_smoke",
            correctness="pass",
            proxy_activity_submitted_work_count=3,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=12,
            required_proxy_channels=12,
        )

        self.assertEqual(record["classification"], "accepted")

    def test_elastic_ll_pass_without_fallback_is_inconclusive(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="elastic_ll",
            correctness="pass",
            proxy_activity_submitted_work_count=3,
            ll_all_rdma_fallback_count=0,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=2,
            required_proxy_channels=2,
        )

        self.assertEqual(record["classification"], "inconclusive")
        self.assertIn("fallback", record["reason"])

    def test_elastic_ll_pass_with_fallback_is_accepted(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="elastic_ll",
            correctness="pass",
            proxy_activity_submitted_work_count=3,
            ll_all_rdma_fallback_count=7,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=2,
            required_proxy_channels=2,
        )

        self.assertEqual(record["classification"], "accepted")

    def test_correctness_failure_is_failed(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="ht_proxy_smoke",
            correctness="fail",
            proxy_activity_submitted_work_count=3,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=12,
            required_proxy_channels=12,
        )

        self.assertEqual(record["classification"], "failed")

    def test_underprovisioned_channels_are_failed(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            rank=0,
            validation_path="ht_proxy_smoke",
            correctness="pass",
            proxy_activity_submitted_work_count=3,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=4,
            required_proxy_channels=12,
        )

        self.assertEqual(record["classification"], "failed")
        self.assertIn("channels", record["reason"])

    def test_ucx_direct_smoke_accepts_ucx_only(self):
        record = make_ep_proxy_evidence(
            backend="ucx",
            rank=0,
            validation_path="ucx_direct_smoke",
            correctness="pass",
        )

        self.assertEqual(record["classification"], "accepted")

    def test_out_of_band_backend_source_is_inconclusive(self):
        record = make_ep_proxy_evidence(
            backend="proxy",
            backend_source="build_directory_name",
            rank=0,
            validation_path="ht_proxy_smoke",
            correctness="pass",
            proxy_activity_submitted_work_count=3,
            proxy_context_published=True,
            proxy_worker_count=1,
            proxy_channel_count=12,
            required_proxy_channels=12,
        )

        self.assertEqual(record["classification"], "inconclusive")

    def test_invalid_single_node_ht_fallback_is_inconclusive(self):
        record = classify_ep_proxy_evidence(
            {
                "kind": "ep_proxy_evidence_v1",
                "backend": "proxy",
                "backend_source": BACKEND_SOURCE,
                "rank": 0,
                "validation_path": "ht_two_node_rdma",
                "correctness": "pass",
                "proxy_context_published": True,
                "proxy_activity_observed": True,
                "unsupported_single_node_fallback": True,
            }
        )

        self.assertEqual(record["classification"], "inconclusive")

    def test_blocked_reason_is_blocked(self):
        record = classify_ep_proxy_evidence(
            {
                "kind": "ep_proxy_evidence_v1",
                "backend": "proxy",
                "backend_source": BACKEND_SOURCE,
                "rank": 0,
                "validation_path": "ht_proxy_smoke",
                "correctness": "not_run",
                "blocked_reason": "CUDA device is unavailable",
            }
        )

        self.assertEqual(record["classification"], "blocked")


if __name__ == "__main__":
    unittest.main()
