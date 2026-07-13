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
      last_seen_gen_(assigned_channel_count, 0) {}

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
        submitReady(channel);
    }
    driveBackendProgress();
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        publishCompletions(channel);
    }
}

void
ProxyWorker::submitReady(nixlProxyChannelState &channel) {
    // Post every newly-produced record to the backend, advancing only the
    // host-side submit cursor. CI (consumer_idx_host_) is deliberately NOT
    // advanced here — it advances solely on completion in publishCompletions(),
    // so each op lives in its ring slot until its network completion arrives.
    for (;;) {
        // consumer_idx is the completion cursor (also read by the GPU for
        // enqueue backpressure); submit_idx_ is host-only (worker is sole
        // accessor). Relaxed loads suffice: the worker is the sole writer of both.
        const uint64_t consumer_idx =
            __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
        const uint64_t submit_idx = channel.submit_idx_;
        // Never submit more than one ring's worth ahead of completions. This is
        // redundant with the op_idx zeroing below (a full ring reads op_idx == 0
        // at the submit slot) but is a cheap, explicit guard against runaway
        // submission and keeps the in-flight set provably bounded by ring depth.
        if (submit_idx - consumer_idx >= channel.ring_depth_) {
            break;
        }
        const uint32_t slot = static_cast<uint32_t>(submit_idx % channel.ring_depth_);
        // op_idx is the GPU-to-CPU signal that the record is written (pairs with
        // the release store in device enqueue). No producer index read on host.
        const uint64_t op_idx =
            __atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE);
        if (op_idx == 0) {
            break;  // nothing new produced at the submit frontier
        }

        nixlProxySubmission submission = channel.records_host_[slot];
        submission.op_idx = op_idx;

        // Zero op_idx now, before advancing submit_idx_. This is load-bearing,
        // not tidiness: the record stays resident in its slot (CI has not moved),
        // so when submit_idx_ later wraps back to this slot at submit_idx +
        // ring_depth — reachable only when the ring is full of in-flight ops
        // (consumer_idx == submit_idx) — the stale op_idx == ticket+1 would read
        // as "ready" and duplicate-submit this ticket. Zeroing makes the slot read
        // 0 so the scan stops. Race-free: the GPU cannot rewrite this slot until
        // CI passes this ticket, which only happens at completion, strictly later.
        __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
        channel.submit_idx_ = submit_idx + 1;

        NIXL_DEBUG << "ProxyWorker::submitReady: channel=" << channel.device_view.channel_id
                   << " submit_idx=" << submit_idx
                   << " opcode=" << static_cast<int>(submission.opcode)
                   << " op_idx=" << submission.op_idx
                   << " size=" << submission.size;

        submitToBackend(channel, slot, submission);
    }
}

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                             uint32_t slot,
                             const nixlProxySubmission &submission) {
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    inflight.opcode = submission.opcode;

    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx
                   << " status=" << status;
        // Terminal error, no backend request. Recorded in the slot so
        // publishCompletions() publishes it in posting order.
        inflight.status = status;
    } else {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
                   << " opcode=" << static_cast<int>(submission.opcode)
                   << " channel=" << submission.channel_id
                   << " local_addr=0x" << std::hex << prepared_submission.local.desc.addr
                   << " remote_addr=0x" << prepared_submission.remote.desc.addr << std::dec
                   << " size=" << submission.size;

        uint64_t request_token = 0;
        if (submitted_work_count_ != nullptr) {
            submitted_work_count_->fetch_add(1, std::memory_order_relaxed);
        }
        status = backend_->submit(prepared_submission, request_token);
        inflight.backend_req_token = request_token;
        // Terminal at submit time (NIXL_SUCCESS or failure) is published without
        // ever polling the backend; NIXL_IN_PROG keeps the default status.
        if (status != NIXL_IN_PROG) {
            inflight.status = status;
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "ProxyWorker::submitToBackend: backend submit failed"
                           << " status=" << status << " op_idx=" << submission.op_idx
                           << " request_token=" << request_token;
            }
        }
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submitted op_idx=" << submission.op_idx
                   << " request_token=" << request_token << " status=" << status;
    }

    channel.inflight_slots_[slot] = inflight;
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
    // Drain completions in posting (FIFO) order over the in-flight window
    // [consumer_idx, submit_idx_). Each completed op advances CI by one, freeing
    // its ring slot for the GPU producer's backpressure check.
    for (;;) {
        // Worker is the sole writer of both cursors — relaxed load suffices.
        const uint64_t consumer_idx =
            __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
        if (consumer_idx >= channel.submit_idx_) {
            break;  // no submitted-but-uncompleted ops
        }
        const uint32_t slot = static_cast<uint32_t>(consumer_idx % channel.ring_depth_);
        nixlProxyRequestState &front = channel.inflight_slots_[slot];
        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backend_req_token);
            if (st == NIXL_IN_PROG) {
                break;  // head-of-line: preserve posting-order completion
            }
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: channel="
                   << channel.device_view.channel_id
                   << " op_idx=" << front.op_idx
                   << " status=" << st
                   << " token=" << front.backend_req_token;
        // Publish completion to the GPU (collapsed CQ), then advance CI so the
        // slot becomes reclaimable. Note front.op_idx == consumer_idx + 1.
        channel.completion_slot_host_->next_status = st;
        __atomic_store_n(&channel.completion_slot_host_->completed_idx,
                         front.op_idx, __ATOMIC_RELEASE);
        __atomic_store_n(channel.consumer_idx_host_, consumer_idx + 1, __ATOMIC_RELEASE);
        if (st != NIXL_SUCCESS) {
            channel.error_latched = true;
            break;
        }
    }
}

void
ProxyWorker::resetChannel(nixlProxyChannelState &channel) {
    // Release any backend requests still in flight for this channel (cancel+free).
    // The in-flight set is the window [consumer_idx, submit_idx_): those records
    // were submitted (op_idx already zeroed) but never completed.
    const uint64_t consumer = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    for (uint64_t idx = consumer; idx < channel.submit_idx_; ++idx) {
        const uint32_t slot = static_cast<uint32_t>(idx % channel.ring_depth_);
        nixlProxyRequestState &inflight = channel.inflight_slots_[slot];
        if (inflight.status == NIXL_IN_PROG && inflight.backend_req_token != 0) {
            backend_->releaseRequest(inflight.backend_req_token);
        }
        inflight = nixlProxyRequestState{};
    }

    // Discard any stale ring entries the previous incarnation produced but that
    // were never submitted (from the submit frontier forward — earlier slots are
    // in-flight and already have op_idx == 0). Draining them normally would hit a
    // retired memview (prepareSubmission -> NIXL_ERR_NOT_FOUND) and immediately
    // re-latch the channel, so we drop them. Safe because the ring is quiescent.
    uint64_t frontier = channel.submit_idx_;
    uint32_t discarded = 0;
    for (;;) {
        const uint32_t slot = static_cast<uint32_t>(frontier % channel.ring_depth_);
        if (__atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE) == 0) {
            break;
        }
        __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
        frontier += 1;
        discarded += 1;
    }

    // Collapse both cursors to the drained frontier and free every slot to the
    // GPU. Monotonic indices (producer/consumer/completed_idx) advance, never rewind.
    channel.submit_idx_ = frontier;
    __atomic_store_n(channel.consumer_idx_host_, frontier, __ATOMIC_RELEASE);

    // Clear the latch and the terminal status so the rank can use the lane again.
    channel.error_latched = false;
    if (channel.completion_slot_host_ != nullptr) {
        channel.completion_slot_host_->next_status = NIXL_IN_PROG;
    }

    NIXL_DEBUG << "ProxyWorker::resetChannel: channel=" << channel.device_view.channel_id
               << " discarded=" << discarded;
}
