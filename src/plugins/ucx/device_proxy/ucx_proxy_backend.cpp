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

#include <algorithm>
#include <sstream>

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

size_t
nixlUcxProxyBackendAdapter::workerIdForChannel(uint32_t channel_id) const {
    const size_t num_workers = engine_->getSharedWorkersSize();
    return num_workers ? (channel_id % num_workers) : 0;
}

std::string
nixlUcxProxyBackendAdapter::workerSubmitHistogram() const {
    if (!stall_log_enabled_ || engine_ == nullptr) {
        return {};
    }
    const size_t num_workers =
        std::min<size_t>(engine_->getSharedWorkersSize(), kMaxTrackedWorkers);
    std::ostringstream oss;
    for (size_t w = 0; w < num_workers; ++w) {
        if (w != 0) {
            oss << ' ';
        }
        oss << 'w' << w << '=' << worker_submit_counts_[w].load(std::memory_order_relaxed);
    }
    return oss.str();
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      uint64_t &request_token) {
    const size_t worker_id = workerIdForChannel(submission.channel_id);
    if (stall_log_enabled_ && worker_id < kMaxTrackedWorkers) {
        worker_submit_counts_[worker_id].fetch_add(1, std::memory_order_relaxed);
    }

    nixlUcxReq req = nullptr;
    const nixl_status_t status = engine_->submitProxyRmaWrite(submission.local.desc,
                                                              submission.remote.desc,
                                                              submission.size,
                                                              worker_id,
                                                              req);
    if (status == NIXL_IN_PROG) {
        request_token = encodeToken(worker_id, req);
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
    const size_t worker_id = workerIdForChannel(submission.channel_id);
    if (stall_log_enabled_ && worker_id < kMaxTrackedWorkers) {
        worker_submit_counts_[worker_id].fetch_add(1, std::memory_order_relaxed);
    }

    nixlUcxReq req = nullptr;
    const nixl_status_t status = engine_->submitProxyAtomicAdd(submission.remote.desc,
                                                               submission.value,
                                                               worker_id,
                                                               req);
    if (status == NIXL_IN_PROG) {
        request_token = encodeToken(worker_id, req);
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
nixlUcxProxyBackendAdapter::shutdown() {
    // Nothing tracked here: outstanding tokens live in the channels' inflight
    // queues and are released by the runtime before it shuts the adapter down.
    return NIXL_SUCCESS;
}
