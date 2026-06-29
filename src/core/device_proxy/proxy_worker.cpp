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
#include <cstdlib>
#include <cuda_runtime.h>

ProxyWorker::ProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                         const nixlProxyMemViewRegistry *proxy_memview_registry,
                         uint32_t *shutdown_word,
                         nixlProxyChannelState *assigned_channels,
                         uint32_t assigned_channel_count,
                         uint64_t pthr_delay_us,
                         std::atomic<uint64_t> *submitted_work_count,
                         std::atomic<uint64_t> *assigned_generations) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      assigned_channels_(assigned_channels),
      assigned_channel_count_(assigned_channel_count),
      pthr_delay_us_(pthr_delay_us),
      submitted_work_count_(submitted_work_count),
      assigned_generations_(assigned_generations),
      last_seen_gen_(assigned_channel_count, 0) {
    stall_log_enabled_ = std::getenv("NIXL_EP_PROXY_STALL_LOG") != nullptr;
    last_stall_log_ = std::chrono::steady_clock::now();
}

ProxyWorker::~ProxyWorker() {
    join();
}

void
ProxyWorker::start(uint32_t worker_idx) {
    thread_ = std::thread([this, worker_idx]() {
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " started";
        while (__atomic_load_n(shutdown_word_, __ATOMIC_ACQUIRE)
               == static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING)) {
            runOnce();
            if (pthr_delay_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pthr_delay_us_));
            }
        }
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " exiting";
    });
}

void
ProxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
ProxyWorker::runOnce() {
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        // Revive: the connection layer bumped this band's generation because its remote
        // agent changed ((re)connect/disconnect). Reconcile lazily here on the worker
        // thread (the sole mutator of channel state). The band is quiescent at the bump
        // (disconnect masks+syncs first; a (re)connecting rank isn't sent to until connect
        // completes) and resetChannel is acquire/release-safe vs a concurrent enqueue
        // regardless, so discarding stale entries cannot corrupt the new incarnation.
        if (assigned_generations_ != nullptr) {
            const uint64_t gen = assigned_generations_[i].load(std::memory_order_acquire);
            if (gen != last_seen_gen_[i]) {
                resetChannel(channel);
                last_seen_gen_[i] = gen;
            }
        }
        nixlProxySubmission submission;
        while (tryDequeue(channel, submission)) {
            submitToBackend(channel, submission);
        }
    }
    driveBackendProgress();
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        publishCompletions(channel);
    }
    maybeLogStalls();
}

void
ProxyWorker::maybeLogStalls() {
    if (!stall_log_enabled_) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_stall_log_ < std::chrono::seconds(1)) {
        return;
    }
    last_stall_log_ = now;
    // Surface any assigned channel that still has outstanding (unpublished) work. A survivor
    // channel showing a growing inflight count with an IN_PROG head => transport stall on the
    // shared worker; an idle survivor channel => the stall is upstream (GPU enqueue / receiver).
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        const nixlProxyChannelState &channel = assigned_channels_[i];
        if (channel.inflight_requests.empty() && !channel.error_latched) {
            continue;
        }
        NIXL_INFO << "ProxyWorker STALLDBG: channel=" << channel.device_view.channel_id
                  << " inflight=" << channel.inflight_requests.size()
                  << " error_latched=" << channel.error_latched
                  << " head_op="
                  << (channel.inflight_requests.empty()
                          ? 0
                          : channel.inflight_requests.front().op_idx)
                  << " head_status="
                  << (channel.inflight_requests.empty()
                          ? 0
                          : static_cast<int>(channel.inflight_requests.front().status));
    }
}

bool
ProxyWorker::tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission) {
    // Sole writer of consumer_idx on host — relaxed load is sufficient.
    uint64_t local_consumer_idx =
        __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t slot = static_cast<uint32_t>(local_consumer_idx % channel.ring_depth_);
    // op_idx is the GPU-to-CPU signal that the record is written
    // (pairs with release store in device enqueue).  No producer index
    // read on host — it is GPU-internal for slot allocation.
    const uint64_t op_idx = __atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return false;
    }
    submission = channel.records_host_[slot];
    submission.op_idx = op_idx;
    __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
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

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel, const nixlProxySubmission &submission) {
    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx
                   << " status=" << status;
        channel.inflight_requests.push_back(
            {submission.op_idx, 0, status});
        // The terminal error is queued for publishCompletions(); the worker handled it.
        return;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id
               << " local_addr=0x" << std::hex << prepared_submission.local.desc.addr
               << " remote_addr=0x" << prepared_submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remote_agent << "'";

    uint64_t request_token = 0;
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    if (submitted_work_count_ != nullptr) {
        submitted_work_count_->fetch_add(1, std::memory_order_relaxed);
    }
    status = backend_->submit(prepared_submission, request_token);
    inflight.backend_req_token = request_token;
    if (status != NIXL_SUCCESS) {
        // backend submit failed, so status is already terminal and can be
        // published without polling the backend.
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
    backend_->progress();
}

void
ProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    if (channel.error_latched) {
        return;
    }
    while (!channel.inflight_requests.empty()) {
        nixlProxyRequestState &front = channel.inflight_requests.front();
        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(channel.device_view.channel_id,
                                           front.backend_req_token);
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

void
ProxyWorker::resetChannel(nixlProxyChannelState &channel) {
    // Discard any stale ring entries left by the previous incarnation of this rank.
    // Draining them normally would hit a retired memview (prepareSubmission ->
    // NIXL_ERR_NOT_FOUND) and immediately re-latch the channel, so we drop them
    // instead of submitting. Safe because the ring is quiescent during reset.
    uint64_t consumer = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t discarded = 0;
    for (;;) {
        const uint32_t slot = static_cast<uint32_t>(consumer % channel.ring_depth_);
        if (__atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE) == 0) {
            break;
        }
        __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
        consumer += 1;
        discarded += 1;
    }
    __atomic_store_n(channel.consumer_idx_host_, consumer, __ATOMIC_RELEASE);

    // Best-effort release of any backend handles still tracked for this channel, then
    // drop the inflight queue. A handle that never reaches a terminal status (rare,
    // transport-failure case) is dropped; the backend adapter cleans up on shutdown.
    for (auto &inflight : channel.inflight_requests) {
        if (inflight.status == NIXL_IN_PROG) {
            backend_->checkCompletion(channel.device_view.channel_id,
                                      inflight.backend_req_token);
        }
    }
    channel.inflight_requests.clear();

    // Clear the latch and the terminal status so the rank can use the lane again.
    // Monotonic indices (producer/consumer/completed_idx) are intentionally left as-is.
    channel.error_latched = false;
    if (channel.completion_slot_host_ != nullptr) {
        channel.completion_slot_host_->next_status = NIXL_IN_PROG;
    }

    NIXL_DEBUG << "ProxyWorker::resetChannel: channel=" << channel.device_view.channel_id
               << " discarded=" << discarded;
}
