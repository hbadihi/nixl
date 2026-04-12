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
#include "proxy_worker.h"
#include "proxy_runtime.h"
#include "backend_adapter.h"
#include "nixl_log.h"
#include <cuda_runtime.h>

ProxyWorker::ProxyWorker(DeviceProxyBackendAdapter *backend,
                         const ProxyMemViewRegistry *proxy_memview_registry,
                         std::atomic<uint32_t> *shutdown_word,
                         ChannelState *assigned_channels,
                         uint32_t assigned_channel_count) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      assigned_channels_(assigned_channels),
      assigned_channel_count_(assigned_channel_count) {}

void
ProxyWorker::runOnce() {
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        ChannelState &channel = assigned_channels_[i];
        ProxySubmission submission;
        while (tryDequeue(channel, submission)) {
            nixl_status_t status = dispatch(channel, submission);
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "ProxyWorker::runOnce: channel=" << channel.device_view.channel_id
                           << " submission failed op_idx=" << submission.op_idx
                           << " status=" << status;
                // continue to the next operation
            }
        }
    }
    driveBackendProgress();
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        ChannelState &channel = assigned_channels_[i];
        publishCompletions(channel);
    }
}

bool
ProxyWorker::tryDequeue(ChannelState &channel, ProxySubmission &submission) {
    WorkRing *ring = channel.device_view.work_ring;
    if (channel.consumer_idx_host_ == nullptr || ring->producer_idx == nullptr) {
        return false;
    }
    uint32_t observed_producer_idx = 0;
    if (cudaMemcpy(&observed_producer_idx,
                   ring->producer_idx,
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost)
        != cudaSuccess) {
        return false;
    }
    uint32_t local_consumer_idx =
        __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_ACQUIRE);
    if (local_consumer_idx == observed_producer_idx) {
        return false;
    }
    submission = ring->records[local_consumer_idx % ring->depth];
    __atomic_store_n(channel.consumer_idx_host_,
                     local_consumer_idx + 1,
                     __ATOMIC_RELEASE);
    return true;
}

nixl_status_t
ProxyWorker::dispatch(ChannelState &channel, const ProxySubmission &submission) {
    nixlMemViewH src_memview = nullptr;
    nixlMemViewH dst_memview = nullptr;
    const auto resolve_or_fallback = [this](uint64_t proxy_memview_id,
                                            nixlMemViewH &out_memview) -> bool {
        if (proxy_memview_registry_->resolveProxyMemViewId(proxy_memview_id, out_memview)) {
            return true;
        }
        // Fallback for phase-1 wiring: allow direct backend memview handles
        // encoded as proxy IDs when explicit proxy registration was not used.
        out_memview = reinterpret_cast<nixlMemViewH>(proxy_memview_id);
        return out_memview != nullptr;
    };

    if (!resolve_or_fallback(submission.dst_proxy_memview_id, dst_memview)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if ((submission.opcode == ProxyOpcode::PUT)
        && !resolve_or_fallback(submission.src_proxy_memview_id, src_memview)) {
        return NIXL_ERR_NOT_FOUND;
    }

    ResolvedProxySubmission resolved_submission = {
        .op_idx = submission.op_idx,
        .opcode = submission.opcode,
        .channel_id = submission.channel_id,
        .flags = submission.flags,
        .src_memview = src_memview,
        .src_index = submission.src_index,
        .src_offset = submission.src_offset,
        .dst_memview = dst_memview,
        .dst_index = submission.dst_index,
        .dst_offset = submission.dst_offset,
        .size = submission.size,
        .value = submission.value,
    };

    uint64_t request_token = 0;
    nixl_status_t status = backend_->submit(resolved_submission, request_token);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    inflight.backend_req_token = request_token;
    inflight.status = NIXL_IN_PROG;
    inflight.publish_ready = false;
    channel.inflight_requests.push_back(inflight);
    return NIXL_SUCCESS;
}

void
ProxyWorker::driveBackendProgress() {
    backend_->progress();
}

void
ProxyWorker::publishCompletions(ChannelState &channel) {
    while (!channel.inflight_requests.empty()) {
        ProxyRequestState &front = channel.inflight_requests.front();
        nixl_status_t st = backend_->checkCompletion(front.backend_req_token);
        if (st == NIXL_IN_PROG) {
            break;
        }
        CompletionSlot slot{};
        slot.completed_idx = front.op_idx;
        slot.next_status = st;
        if (cudaMemcpy(channel.device_view.completion_slot,
                       &slot,
                       sizeof(CompletionSlot),
                       cudaMemcpyHostToDevice)
            != cudaSuccess) {
            break;
        }
        channel.inflight_requests.erase(channel.inflight_requests.begin());
    }
}
