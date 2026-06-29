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
constexpr uint64_t kInvalidToken = 0;
}

nixl_status_t
nixlUcxProxyBackendAdapter::init(uint32_t, uint32_t channel_count) {
    tracked_requests_.clear();
    tracked_requests_.resize(channel_count);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submit(const nixlBackendProxySubmission &submission,
                                   uint64_t &request_token) {
    request_token = kInvalidToken;
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

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      uint64_t &request_token) {
    if (submission.channel_id >= tracked_requests_.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t worker_id = workerIdForChannel(submission.channel_id);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->submitProxyRmaWrite(submission.local.desc,
                                                        submission.remote.desc,
                                                        submission.size,
                                                        worker_id,
                                                        handle);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: submitProxyRmaWrite failed "
                      "status="
                   << status;
        return status;
    }

    request_token = trackRequest(submission.channel_id, submission.op_idx, handle);
    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex
               << submission.local.desc.addr << std::dec
               << " dst_addr=0x" << std::hex
               << submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " remote_agent='" << submission.remote_agent << "'"
               << " token=" << request_token;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitAtomicAdd(const nixlBackendProxySubmission &submission,
                                            uint64_t &request_token) {
    if (submission.channel_id >= tracked_requests_.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Same channel -> worker mapping as submitPut so a channel's put and its follow-up
    // atomic flag travel the same worker/EP/QP, preserving IB write-before-atomic order.
    const size_t worker_id = workerIdForChannel(submission.channel_id);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->submitProxyAtomicAdd(submission.remote.desc,
                                                         submission.value,
                                                         worker_id,
                                                         handle);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: submitProxyAtomicAdd "
                      "failed status="
                   << status;
        return status;
    }

    request_token = trackRequest(submission.channel_id, submission.op_idx, handle);
    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: posted RDMA atomic add"
               << " dst_addr=0x" << std::hex
               << submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " value=" << submission.value
               << " remote_agent='" << submission.remote_agent << "'"
               << " token=" << request_token;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::checkCompletion(uint32_t channel_id, uint64_t request_token) {
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (channel_id >= tracked_requests_.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto &channel_requests = tracked_requests_[channel_id];
    if (channel_requests.empty()) {
        return NIXL_ERR_NOT_FOUND;
    }

    auto it = channel_requests.begin();
    if (it->op_idx != request_token) {
        // resetChannel() can drop proxy-side inflight state while older backend
        // handles remain queued for shutdown cleanup. Let revived channels find
        // their new request without being blocked by those stale entries.
        for (; it != channel_requests.end(); ++it) {
            if (it->op_idx == request_token) {
                break;
            }
        }
        if (it == channel_requests.end()) {
            return NIXL_ERR_NOT_FOUND;
        }
    }

    nixlBackendReqH *handle = it->handle;
    const nixl_status_t status = engine_->checkProxyReqStatus(handle);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    channel_requests.erase(it);

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::checkCompletion: channel=" << channel_id
               << " token=" << request_token
               << " status=" << status;
    engine_->releaseReqH(handle);
    return status;
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
    size_t tracked_request_count = 0;
    for (const auto &channel_requests : tracked_requests_) {
        tracked_request_count += channel_requests.size();
    }

    NIXL_INFO << "nixlUcxProxyBackendAdapter::shutdown: releasing "
              << tracked_request_count << " tracked request(s)";
    if (engine_ != nullptr) {
        for (auto &channel_requests : tracked_requests_) {
            for (auto &request : channel_requests) {
                engine_->releaseReqH(request.handle);
            }
        }
    }
    for (auto &channel_requests : tracked_requests_) {
        channel_requests.clear();
    }
    return NIXL_SUCCESS;
}

uint64_t
nixlUcxProxyBackendAdapter::trackRequest(uint32_t channel_id,
                                         uint64_t op_idx,
                                         nixlBackendReqH *handle) {
    NIXL_ASSERT(channel_id < tracked_requests_.size());
    tracked_requests_[channel_id].push_back({op_idx, handle});
    return op_idx;
}
