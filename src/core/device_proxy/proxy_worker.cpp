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
                         uint32_t *shutdown_word,
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
    if (ring == nullptr || channel.consumer_idx_host_ == nullptr) {
        return false;
    }
    // Sole writer of consumer_idx on host — relaxed load is sufficient.
    uint32_t local_consumer_idx =
        __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t slot = local_consumer_idx % ring->depth;
    // ready_flag is the GPU-to-CPU signal that the record is written
    // (pairs with release store in device enqueue).  No producer_idx
    // read on host — it is GPU-internal for slot allocation.
    if (!__atomic_load_n(&ring->records[slot].ready_flag, __ATOMIC_ACQUIRE)) {
        return false;
    }
    submission = ring->records[slot];
    __atomic_store_n(&ring->records[slot].ready_flag, 0, __ATOMIC_RELAXED);
    __atomic_store_n(channel.consumer_idx_host_,
                     local_consumer_idx + 1,
                     __ATOMIC_RELEASE);
    NIXL_DEBUG << "ProxyWorker::tryDequeue: channel=" << channel.device_view.channel_id
               << " consumer=" << local_consumer_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " op_idx=" << submission.op_idx
               << " size=" << submission.size;
    return true;
}

nixl_status_t
ProxyWorker::dispatch(ChannelState &channel, const ProxySubmission &submission) {
    nixlMemViewH src_memview = nullptr;
    nixlMemViewH dst_memview = nullptr;
    if (!proxy_memview_registry_->resolveProxyMemViewId(submission.dst_proxy_memview_id,
                                                        dst_memview)) {
        NIXL_DEBUG << "ProxyWorker::dispatch: dst memview resolution failed"
                   << " dst_proxy_id=" << submission.dst_proxy_memview_id;
        channel.inflight_requests.push_back(
            {submission.op_idx, 0, NIXL_ERR_NOT_FOUND, true});
        return NIXL_ERR_NOT_FOUND;
    }
    if ((submission.opcode == ProxyOpcode::PUT)
        && !proxy_memview_registry_->resolveProxyMemViewId(
            submission.src_proxy_memview_id, src_memview)) {
        NIXL_DEBUG << "ProxyWorker::dispatch: src memview resolution failed"
                   << " src_proxy_id=" << submission.src_proxy_memview_id;
        channel.inflight_requests.push_back(
            {submission.op_idx, 0, NIXL_ERR_NOT_FOUND, true});
        return NIXL_ERR_NOT_FOUND;
    }

    NIXL_DEBUG << "ProxyWorker::dispatch: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id
               << " src_mvh=" << src_memview << " dst_mvh=" << dst_memview
               << " src_off=" << submission.src_offset
               << " dst_off=" << submission.dst_offset
               << " size=" << submission.size;

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
    ProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    nixl_status_t status = backend_->submit(resolved_submission, request_token);
    inflight.backend_req_token = request_token;
    if (status != NIXL_SUCCESS) {
        // backend submit failed, set the status and publish_ready to true
        NIXL_ERROR << "ProxyWorker::dispatch: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.op_idx
                   << " request_token=" << request_token;
        inflight.status = status;
        inflight.publish_ready = true;
    } else {
        status = NIXL_IN_PROG;
        inflight.publish_ready = false;
    }

    NIXL_DEBUG << "ProxyWorker::dispatch: submitted op_idx=" << submission.op_idx
               << " request_token=" << request_token << " status=" << status;
    channel.inflight_requests.push_back(inflight);
    return NIXL_SUCCESS;
}

void
ProxyWorker::driveBackendProgress() {
    backend_->progress();
}

void
ProxyWorker::publishCompletions(ChannelState &channel) {
    if (channel.error_latched) {
        return;
    }
    while (!channel.inflight_requests.empty()) {
        ProxyRequestState &front = channel.inflight_requests.front();
        nixl_status_t st;
        if (front.publish_ready) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backend_req_token);
            if (st == NIXL_IN_PROG) {
                break;
            }
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: channel="
                   << channel.device_view.channel_id
                   << " op_idx=" << front.op_idx
                   << " status=" << st
                   << " token=" << front.backend_req_token;
        channel.completion_slot_host_->next_status = st;
        __atomic_store_n(&channel.completion_slot_host_->completed_idx,
                         front.op_idx, __ATOMIC_RELEASE);
        channel.inflight_requests.pop_front();
        if (st != NIXL_SUCCESS) {
            channel.error_latched = true;
            break;
        }
    }
}
