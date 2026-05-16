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
#include <cstdio>
#include <cuda_runtime.h>

namespace {
uint64_t
steadyClockNs() noexcept {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
} // namespace

void
ProxyWorker::TimingStats::record(uint64_t duration_ns) noexcept {
    ++count;
    sum_ns += duration_ns;
    if (duration_ns < min_ns) {
        min_ns = duration_ns;
    }
    if (duration_ns > max_ns) {
        max_ns = duration_ns;
    }
}

ProxyWorker::ProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                         const nixlProxyMemViewRegistry *proxy_memview_registry,
                         uint32_t *shutdown_word,
                         nixlProxyChannelState *assigned_channels,
                         uint32_t assigned_channel_count,
                         uint64_t pthr_delay_us) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      assigned_channels_(assigned_channels),
      assigned_channel_count_(assigned_channel_count),
      pthr_delay_us_(pthr_delay_us) {}

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
        printStats(worker_idx);
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
    ++run_once_iters_;
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
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
}

bool
ProxyWorker::tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission) {
    // Sole writer of consumer_idx on host — relaxed load is sufficient.
    uint64_t local_consumer_idx =
        __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t slot = static_cast<uint32_t>(local_consumer_idx % channel.ring_depth_);
    // op_idx is the GPU-to-CPU signal that the record is written
    // (pairs with release store in device enqueue).  No producer ticket
    // read on host — it is GPU-internal for slot allocation.
    const uint64_t op_idx = __atomic_load_n(&channel.records_[slot].op_idx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return false;
    }
    submission = channel.records_[slot];
    submission.op_idx = op_idx;
    __atomic_store_n(&channel.records_[slot].op_idx, 0, __ATOMIC_RELAXED);
    __atomic_store_n(channel.consumer_idx_host_,
                     local_consumer_idx + 1,
                     __ATOMIC_RELEASE);
    __atomic_store_n(channel.dequeued_idx_host_, submission.op_idx, __ATOMIC_RELEASE);
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
    const uint64_t prepare_start_ns = steadyClockNs();
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    const uint64_t prepare_end_ns = steadyClockNs();
    prepare_stats_.record(prepare_end_ns - prepare_start_ns);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx
                   << " status=" << status;
        channel.inflight_requests.push_back(
            {submission.op_idx, 0, status});
        // The terminal error is queued for publishCompletions(); the worker handled it.
        return;
    }
    __atomic_store_n(channel.prepared_idx_host_, submission.op_idx, __ATOMIC_RELEASE);

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id
               << " local_addr=0x" << std::hex << prepared_submission.local.desc.addr
               << " remote_addr=0x" << prepared_submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " remote_agent='"
               << (prepared_submission.remote_agent ? prepared_submission.remote_agent->c_str() : "<null>")
               << "'";

    uint64_t request_token = 0;
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    const uint64_t submit_start_ns = steadyClockNs();
    status = backend_->submit(prepared_submission, request_token);
    const uint64_t submit_end_ns = steadyClockNs();
    submit_stats_.record(submit_end_ns - submit_start_ns);
    __atomic_store_n(channel.submitted_idx_host_, submission.op_idx, __ATOMIC_RELEASE);
    inflight.backend_req_token = request_token;
    inflight.submit_done_ns = submit_end_ns;
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
    ++progress_calls_;
    bool has_inflight = false;
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        if (!assigned_channels_[i].inflight_requests.empty()) {
            has_inflight = true;
            break;
        }
    }
    if (!has_inflight) {
        backend_->progress();
        return;
    }

    const uint64_t progress_start_ns = steadyClockNs();
    backend_->progress();
    const uint64_t progress_ns = steadyClockNs() - progress_start_ns;
    progress_stats_.record(progress_ns);

    // Attribute progress time to the request currently eligible to complete on
    // each channel. This keeps the common single-inflight pingpong case exact.
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        if (!channel.inflight_requests.empty()) {
            channel.inflight_requests.front().progress_ns += progress_ns;
        }
    }
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
            ++front.completion_polls;
            const uint64_t check_start_ns = steadyClockNs();
            st = backend_->checkCompletion(front.backend_req_token);
            const uint64_t check_ns = steadyClockNs() - check_start_ns;
            front.check_completion_ns += check_ns;
            check_completion_stats_.record(check_ns);
            if (st == NIXL_IN_PROG) {
                break;
            }
        }
        const uint64_t terminal_status_ns = steadyClockNs();
        if (front.submit_done_ns != 0) {
            const uint64_t post_submit_ns = terminal_status_ns - front.submit_done_ns;
            post_submit_stats_.record(post_submit_ns);
            post_submit_progress_stats_.record(front.progress_ns);
            post_submit_check_stats_.record(front.check_completion_ns);
            const uint64_t active_ns = front.progress_ns + front.check_completion_ns;
            post_submit_wait_stats_.record(
                active_ns < post_submit_ns ? post_submit_ns - active_ns : 0);
            total_completion_polls_ += front.completion_polls;
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: channel="
                   << channel.device_view.channel_id
                   << " op_idx=" << front.op_idx
                   << " status=" << st
                   << " token=" << front.backend_req_token;
        const uint64_t publish_start_ns = steadyClockNs();
        channel.completion_slot_host_->next_status = st;
        __atomic_store_n(&channel.completion_slot_host_->completed_idx,
                         front.op_idx, __ATOMIC_RELEASE);
        publish_stats_.record(steadyClockNs() - publish_start_ns);
        channel.inflight_requests.pop_front();
        if (st != NIXL_SUCCESS) {
            channel.error_latched = true;
            break;
        }
    }
}

void
ProxyWorker::printStats(uint32_t worker_idx) const noexcept {
    auto print_timing = [worker_idx](const char *name, const TimingStats &stats) {
        const double avg_us = stats.count == 0 ? 0.0 :
            static_cast<double>(stats.sum_ns) / static_cast<double>(stats.count) / 1000.0;
        const double min_us = stats.count == 0 ? 0.0 :
            static_cast<double>(stats.min_ns) / 1000.0;
        const double max_us = stats.count == 0 ? 0.0 :
            static_cast<double>(stats.max_ns) / 1000.0;
        std::fprintf(stderr,
                     "[proxy-worker-stats][w%u] %-11s n=%llu avg=%9.3f us min=%9.3f us max=%9.3f us\n",
                     worker_idx,
                     name,
                     static_cast<unsigned long long>(stats.count),
                     avg_us,
                     min_us,
                     max_us);
    };

    print_timing("prepare", prepare_stats_);
    print_timing("submit", submit_stats_);
    print_timing("post_submit", post_submit_stats_);
    print_timing("progress", progress_stats_);
    print_timing("check", check_completion_stats_);
    print_timing("post_progress", post_submit_progress_stats_);
    print_timing("post_check", post_submit_check_stats_);
    print_timing("post_wait", post_submit_wait_stats_);
    print_timing("publish", publish_stats_);

    const double polls_per_request = post_submit_stats_.count == 0 ? 0.0 :
        static_cast<double>(total_completion_polls_) / static_cast<double>(post_submit_stats_.count);
    std::fprintf(stderr,
                 "[proxy-worker-stats][w%u] polls/request=%9.3f progress_calls=%llu runOnce_iters=%llu\n",
                 worker_idx,
                 polls_per_request,
                 static_cast<unsigned long long>(progress_calls_),
                 static_cast<unsigned long long>(run_once_iters_));
}
