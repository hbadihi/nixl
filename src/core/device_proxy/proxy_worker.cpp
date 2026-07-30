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
#include <chrono>

ProxyWorker::ProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                         const nixlProxyMemViewRegistry *proxy_memview_registry,
                         uint32_t *shutdown_word,
                         nixlProxyChannelState *channels,
                         uint32_t max_peers,
                         uint32_t channel_count,
                         uint32_t worker_index,
                         uint32_t worker_count,
                         uint64_t pthr_delay_us) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      channels_(channels),
      max_peers_(max_peers),
      channel_count_(channel_count),
      worker_index_(worker_index),
      worker_count_(worker_count),
      pthr_delay_us_(pthr_delay_us) {}

ProxyWorker::~ProxyWorker() {
    join();
}

void
ProxyWorker::start() {
    thread_ = std::thread([this]() {
        NIXL_INFO << "ProxyWorker thread " << worker_index_ << " started";
        while (__atomic_load_n(shutdown_word_, __ATOMIC_ACQUIRE) ==
               static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING)) {
            runOnce();
            if (pthr_delay_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pthr_delay_us_));
            }
        }
        NIXL_INFO << "ProxyWorker thread " << worker_index_ << " exiting";
    });
}

void
ProxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

nixlProxyChannelState *
ProxyWorker::getChannelState(uint32_t peer, uint32_t channel_id) {
    return &channels_[static_cast<size_t>(channel_id) * max_peers_ + peer];
}

void
ProxyWorker::publishOwnedChannels() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            nixlProxyChannelState *channel = getChannelState(peer, channel_id);
            publishCompletions(*channel);
        }
    }
}

void
ProxyWorker::submitOwnedChannels() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            nixlProxyChannelState *channel = getChannelState(peer, channel_id);
            nixlProxySubmission submission;
            while (tryDequeue(*channel, submission)) {
                submitToBackend(*channel, peer, submission);
            }
        }
    }
}

void
ProxyWorker::runOnce() {
    submitOwnedChannels();
    driveBackendProgress();
    publishOwnedChannels();
}

bool
ProxyWorker::tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission) {
    uint64_t local_consumer_idx = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t slot = static_cast<uint32_t>(local_consumer_idx % channel.ring_depth_);
    const uint64_t op_idx = __atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return false;
    }
    submission = channel.records_host_[slot];
    submission.op_idx = op_idx;
    __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
    __atomic_store_n(channel.consumer_idx_host_, local_consumer_idx + 1, __ATOMIC_RELEASE);
    NIXL_DEBUG << "ProxyWorker::tryDequeue: channel=" << submission.channel_id
               << " consumer=" << local_consumer_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " op_idx=" << submission.op_idx << " size=" << submission.size;
    return true;
}

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                             uint32_t peer,
                             const nixlProxySubmission &submission) {
    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    prepared_submission.peer_index = peer;
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx << " status=" << status;
        channel.inflight_requests.push_back({submission.op_idx, 0, status});
        return;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id << " local_addr=0x" << std::hex
               << prepared_submission.local.desc.addr << " remote_addr=0x"
               << prepared_submission.remote.desc.addr << std::dec << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remote_agent << "'";

    uint64_t request_token = 0;
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    status = backend_->submit(prepared_submission, request_token);
    inflight.backend_req_token = request_token;
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyWorker::submitToBackend: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.op_idx
                   << " request_token=" << request_token;
        inflight.status = status;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: submitted op_idx=" << submission.op_idx
               << " request_token=" << request_token << " status=" << status;
    channel.inflight_requests.push_back(inflight);
}

void
ProxyWorker::driveBackendProgress() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            backend_->progress(channel_id, peer);
        }
    }
}

void
ProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    while (!channel.inflight_requests.empty()) {
        nixlProxyRequestState &front = channel.inflight_requests.front();
        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backend_req_token);
            if (st == NIXL_IN_PROG) {
                break;
            }
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: op_idx=" << front.op_idx << " status=" << st
                   << " token=" << front.backend_req_token;

        if (channel.completion_slot_host_->next_status >= 0) {
            channel.completion_slot_host_->next_status = st;
            __atomic_store_n(
                &channel.completion_slot_host_->completed_idx, front.op_idx, __ATOMIC_RELEASE);
        }
        channel.inflight_requests.pop_front();
    }
}
