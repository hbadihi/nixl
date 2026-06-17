# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small UCX-direct correctness smoke for EP selected-backend builds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nixl_ep
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_evidence  # noqa: E402


def run_smoke(args: argparse.Namespace) -> None:
    backend = nixl_ep.get_gpu_device_api_backend()
    correctness = "fail"
    try:
        if backend != "ucx":
            raise RuntimeError(f"UCX-direct smoke requires backend=ucx, got {backend}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for UCX-direct smoke")

        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        torch.cuda.set_device(args.device)

        num_experts = args.num_experts_per_rank
        rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
            args.num_tokens,
            args.hidden_dim,
            1,
            num_experts,
        )
        buffer = nixl_ep.Buffer(
            rank=0,
            low_latency_mode=True,
            explicitly_destroy=True,
            timeout_ms=args.timeout_ms,
        )
        buffer.update_memory_buffers(
            num_ranks=1,
            num_experts_per_rank=args.num_experts_per_rank,
            num_rdma_bytes=rdma_bytes,
        )
        if not buffer.runtime.is_available():
            raise RuntimeError("UCX-direct smoke buffer did not become available")
        buffer.destroy()
        correctness = "pass"
    finally:
        if args.evidence_output:
            record = proxy_evidence.make_ep_proxy_evidence(
                backend=backend,
                rank=0,
                validation_path="ucx_direct_smoke",
                correctness=correctness,
            )
            proxy_evidence.write_evidence_record(record, args.evidence_output)

    if correctness != "pass":
        raise RuntimeError("UCX-direct smoke failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="UCX-direct EP correctness smoke")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-experts-per-rank", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--evidence-output", type=str)
    args = parser.parse_args()
    run_smoke(args)


if __name__ == "__main__":
    main()
