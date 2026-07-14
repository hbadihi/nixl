/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ucx_proxy_backend.h"
#include "../ucx_backend.h"
#include "nixl_log.h"
#include "nixl_types.h"

namespace {
// Token layout: worker_id in the top 8 bits, ucp request pointer in the low 56
// (user-space pointers fit well within 56 bits on x86-64/aarch64). worker_id is
// carried so release/cancel can reach the owning UCX worker without any lookup.
// A token of 0 means "no pending request" (op completed at submit time).
constexpr unsigned kTokenWorkerShift = 56;
constexpr uint64_t kTokenReqMask = (uint64_t{1} << kTokenWorkerShift) - 1;

uint64_t
encodeToken(size_t worker_id, nixlUcxReq req) noexcept {
    return (static_cast<uint64_t>(worker_id) << kTokenWorkerShift) |
        reinterpret_cast<uint64_t>(req);
}

nixlUcxReq
tokenReq(uint64_t token) noexcept {
    return reinterpret_cast<nixlUcxReq>(token & kTokenReqMask);
}

size_t
tokenWorkerId(uint64_t token) noexcept {
    return static_cast<size_t>(token >> kTokenWorkerShift);
}
} // namespace

std::optional<size_t>
nixlUcxProxyWorkerIdForChannel(uint32_t channel_id,
                               size_t num_workers,
                               uint32_t channel_count,
                               const nixlUcxProxyRankMapping &mapping) noexcept {
    const size_t channels_per_rank = mapping.channels_per_rank;
    if (channels_per_rank == 0 || channel_count < channels_per_rank ||
        (channel_count % channels_per_rank) != 0) {
        return std::nullopt;
    }

    const size_t rank_count = channel_count / channels_per_rank;
    const size_t expected_workers =
        (rank_count == 1) ? 1 : (channel_count - channels_per_rank);
    if (num_workers != expected_workers || mapping.local_rank >= rank_count) {
        return std::nullopt;
    }

    const size_t dst_rank = channel_id / channels_per_rank;
    const size_t lane = channel_id % channels_per_rank;
    if (dst_rank >= rank_count || dst_rank == mapping.local_rank) {
        return std::nullopt;
    }

    const size_t compact_dst_rank = dst_rank - (dst_rank > mapping.local_rank);
    const size_t worker_id = compact_dst_rank * channels_per_rank + lane;
    return (worker_id < num_workers) ? std::optional<size_t>(worker_id) : std::nullopt;
}

nixl_status_t
nixlUcxProxyBackendAdapter::init(uint32_t, uint32_t channel_count) {
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!rank_mapping_) {
        channel_count_ = channel_count;
        return NIXL_SUCCESS;
    }

    const size_t channels_per_rank = rank_mapping_->channels_per_rank;
    const size_t num_workers = engine_->getSharedWorkersSize();
    const bool valid_layout =
        channels_per_rank > 0 && channel_count >= channels_per_rank &&
        (channel_count % channels_per_rank) == 0;
    const size_t rank_count =
        valid_layout ? (channel_count / channels_per_rank) : 0;
    const size_t expected_workers =
        !valid_layout ? 0 :
        ((rank_count == 1) ? 1 : (channel_count - channels_per_rank));
    if (!valid_layout ||
        rank_mapping_->local_rank >= rank_count ||
        num_workers != expected_workers) {
        NIXL_ERROR << "nixlUcxProxyBackendAdapter::init: invalid compact rank mapping"
                   << " local_rank=" << rank_mapping_->local_rank
                   << " channels_per_rank=" << channels_per_rank
                   << " channel_count=" << channel_count
                   << " ucx_worker_count=" << num_workers;
        return NIXL_ERR_INVALID_PARAM;
    }

    channel_count_ = channel_count;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submit(const nixlBackendProxySubmission &submission,
                                   uint64_t &request_token) {
    request_token = 0;
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        return submitPut(submission, request_token);
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        return submitAtomicAdd(submission, request_token);
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

std::optional<size_t>
nixlUcxProxyBackendAdapter::workerIdForChannel(uint32_t channel_id) const {
    if (engine_ == nullptr) {
        return std::nullopt;
    }

    const size_t num_workers = engine_->getSharedWorkersSize();
    if (rank_mapping_) {
        return nixlUcxProxyWorkerIdForChannel(
            channel_id, num_workers, channel_count_, *rank_mapping_);
    }

    return num_workers ? std::optional<size_t>(channel_id % num_workers) : std::nullopt;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      uint64_t &request_token) {
    const std::optional<size_t> worker_id = workerIdForChannel(submission.channel_id);
    if (!worker_id) {
        NIXL_ERROR << "nixlUcxProxyBackendAdapter::submitPut: no UCX worker for channel "
                   << submission.channel_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlUcxReq req = nullptr;
    const nixl_status_t status = engine_->submitProxyRmaWrite(submission.local.desc,
                                                              submission.remote.desc,
                                                              submission.size,
                                                              *worker_id,
                                                              req);
    if (status == NIXL_IN_PROG) {
        request_token = encodeToken(*worker_id, req);
    } else if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: submitProxyRmaWrite failed "
                      "status="
                   << status;
        return status;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex
               << submission.local.desc.addr << std::dec
               << " dst_addr=0x" << std::hex
               << submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " token=" << request_token;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitAtomicAdd(const nixlBackendProxySubmission &submission,
                                            uint64_t &request_token) {
    // Same channel -> worker mapping as submitPut so a channel's put and its follow-up
    // atomic flag travel the same worker/EP/QP, preserving IB write-before-atomic order.
    const std::optional<size_t> worker_id = workerIdForChannel(submission.channel_id);
    if (!worker_id) {
        NIXL_ERROR << "nixlUcxProxyBackendAdapter::submitAtomicAdd: no UCX worker for channel "
                   << submission.channel_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlUcxReq req = nullptr;
    const nixl_status_t status = engine_->submitProxyAtomicAdd(submission.remote.desc,
                                                               submission.value,
                                                               *worker_id,
                                                               req);
    if (status == NIXL_IN_PROG) {
        request_token = encodeToken(*worker_id, req);
    } else if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: submitProxyAtomicAdd "
                      "failed status="
                   << status;
        return status;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: posted RDMA atomic add"
               << " dst_addr=0x" << std::hex
               << submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " value=" << submission.value
               << " token=" << request_token;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::checkCompletion(uint64_t request_token) {
    if (engine_ == nullptr || request_token == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlUcxReq req = tokenReq(request_token);
    const nixl_status_t status = engine_->checkProxyRequest(req);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::checkCompletion: token=" << request_token
               << " status=" << status;
    engine_->releaseProxyRequest(tokenWorkerId(request_token), req);
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::releaseRequest(uint64_t request_token) {
    if (engine_ == nullptr || request_token == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    engine_->releaseProxyRequest(tokenWorkerId(request_token), tokenReq(request_token));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress() {
    if (engine_ != nullptr && !progress_thread_enabled_) {
        engine_->progress();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress(uint32_t channel_id) {
    if (engine_ != nullptr && !progress_thread_enabled_) {
        const std::optional<size_t> worker_id = workerIdForChannel(channel_id);
        if (worker_id) {
            engine_->progress(*worker_id);
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::shutdown() {
    // Nothing tracked here: outstanding tokens live in the channels' inflight
    // queues and are released by the runtime before it shuts the adapter down.
    return NIXL_SUCCESS;
}
