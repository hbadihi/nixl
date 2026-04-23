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
#include <cstdlib>

// NVTX 3 is fully header-only — the inline shim lazy-loads the implementation
// at runtime when a profiler attaches and is otherwise a near no-op.  No link
// against libnvToolsExt is required.  Guarded so builds without CUDA headers
// still compile.
#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define NIXL_NVTX_ENABLED 1
#  endif
#endif

#ifdef NIXL_NVTX_ENABLED
namespace {
struct NvtxScopedRange {
    explicit NvtxScopedRange(const char *name) noexcept { nvtxRangePushA(name); }
    ~NvtxScopedRange() noexcept { nvtxRangePop(); }
    NvtxScopedRange(const NvtxScopedRange &) = delete;
    NvtxScopedRange &operator=(const NvtxScopedRange &) = delete;
};
} // namespace
#  define NIXL_NVTX_CONCAT2(a, b) a##b
#  define NIXL_NVTX_CONCAT(a, b)  NIXL_NVTX_CONCAT2(a, b)
#  define NIXL_NVTX_RANGE(name)                                               \
      NvtxScopedRange NIXL_NVTX_CONCAT(_nvtx_range_, __LINE__)(name)
#  define NIXL_NVTX_MARK(name)  nvtxMarkA(name)
#else
#  define NIXL_NVTX_RANGE(name) ((void)0)
#  define NIXL_NVTX_MARK(name)  ((void)0)
#endif

namespace {

using steady_clock = std::chrono::steady_clock;

inline uint64_t
ns_since(const steady_clock::time_point &t0, const steady_clock::time_point &t1) noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

bool
statsEnabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("NIXL_PROXY_STATS");
        // Default ON.  Disable explicitly with: 0, false/False/FALSE,
        // off/Off/OFF, no/No/NO.  Anything else (1, on, true, yes, ...) keeps
        // stats on.  The previous shortcut treated any 'o*' as disable, which
        // accidentally silenced "on"/"On"/"ON" — fixed by checking full prefixes.
        if (env == nullptr || env[0] == '\0') return true;
        auto eq_ci = [](const char *a, const char *b) {
            for (; *a && *b; ++a, ++b) {
                const char ca = (*a >= 'A' && *a <= 'Z') ? char(*a + 32) : *a;
                const char cb = (*b >= 'A' && *b <= 'Z') ? char(*b + 32) : *b;
                if (ca != cb) return false;
            }
            return *a == '\0' && *b == '\0';
        };
        if (env[0] == '0' && env[1] == '\0') return false;
        if (eq_ci(env, "false") || eq_ci(env, "off") || eq_ci(env, "no")) {
            return false;
        }
        return true;
    }();
    return enabled;
}

} // namespace

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

ProxyWorker::~ProxyWorker() {
    join();
}

void
ProxyWorker::start(uint32_t worker_idx) {
    thread_ = std::thread([this, worker_idx]() {
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " started";
        while (__atomic_load_n(shutdown_word_, __ATOMIC_ACQUIRE)
               == static_cast<uint32_t>(ProxyControlState::Running)) {
            runOnce();
        }
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " exiting";
        if (statsEnabled()) {
            logStatsSummary(worker_idx);
        }
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
    ++run_once_count_;
    const bool stats_on = statsEnabled();
    bool any_inflight = false;
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        ChannelState &channel = assigned_channels_[i];
        ProxySubmission submission;
        while (tryDequeue(channel, submission)) {
            // Bracket each real submission so it shows up in nsys/NVTX timelines.
            // No range is emitted for the (overwhelmingly common) empty-poll path,
            // which keeps the report size sane even at hundreds of kpolls/s.
            NIXL_NVTX_RANGE("prx:submit");
            const auto t_dequeue = stats_on ? steady_clock::now() : steady_clock::time_point{};
            nixl_status_t status = submitToBackend(channel, submission);
            if (stats_on) {
                const auto t_submitted = steady_clock::now();
                prep_submit_stats_.record(ns_since(t_dequeue, t_submitted));
                // Stamp the just-pushed inflight request so publishCompletions
                // can compute the inflight latency.
                if (!channel.inflight_requests.empty()) {
                    channel.inflight_requests.back().submit_time = t_submitted;
                }
            }
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "ProxyWorker::runOnce: channel=" << channel.device_view.channel_id
                           << " submission failed op_idx=" << submission.op_idx
                           << " status=" << status;
                // continue to the next operation
            }
        }
        if (!channel.inflight_requests.empty()) {
            any_inflight = true;
        }
    }
    // Only bracket progress when there is in-flight work to drive.  Otherwise
    // the pure-idle spin (millions of empty progress() calls per second) would
    // dwarf every other NVTX event in the capture.
    if (any_inflight) {
        NIXL_NVTX_RANGE("prx:progress");
        driveBackendProgress();
    } else {
        driveBackendProgress();
    }
    ++progress_count_;
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        ChannelState &channel = assigned_channels_[i];
        publishCompletions(channel);
    }
}

bool
ProxyWorker::tryDequeue(ChannelState &channel, ProxySubmission &submission) {
    WorkRing *ring = channel.work_ring_;
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
ProxyWorker::submitToBackend(ChannelState &channel, const ProxySubmission &submission) {
    PreparedProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx
                   << " status=" << status;
        channel.inflight_requests.push_back(
            {submission.op_idx, 0, status});
        return status;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id
               << " local_addr=0x" << std::hex << prepared_submission.local.desc.addr
               << " remote_addr=0x" << prepared_submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remote_agent << "'";

    uint64_t request_token = 0;
    ProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
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
    return NIXL_SUCCESS;
}

void
ProxyWorker::driveBackendProgress() {
    backend_->progress();
}

void
ProxyWorker::logStatsSummary(uint32_t worker_idx) const {
    auto print_stage = [worker_idx](const char *name, const ProxyWorkerStats &s) {
        if (s.count == 0) {
            fprintf(stderr, "[proxy-stats][w%u] %-12s (no samples)\n",
                    worker_idx, name);
            return;
        }
        const double avg_us = (static_cast<double>(s.sum_ns) / s.count) / 1000.0;
        const double min_us = static_cast<double>(s.min_ns) / 1000.0;
        const double max_us = static_cast<double>(s.max_ns) / 1000.0;
        fprintf(stderr,
                "[proxy-stats][w%u] %-12s n=%-8lu avg=%9.3f us  min=%9.3f us  max=%9.3f us\n",
                worker_idx, name,
                static_cast<unsigned long>(s.count), avg_us, min_us, max_us);
    };

    fprintf(stderr,
            "[proxy-stats][w%u] runOnce_iters=%lu  progress_calls=%lu\n",
            worker_idx,
            static_cast<unsigned long>(run_once_count_),
            static_cast<unsigned long>(progress_count_));
    print_stage("prep+submit", prep_submit_stats_);   // dequeue → submit ret
    print_stage("inflight",    inflight_stats_);      // submit ret → completion seen
    print_stage("publish",     publish_stats_);       // completion → published
    if (prep_submit_stats_.count > 0) {
        const double polls_per_req =
            static_cast<double>(run_once_count_) /
            static_cast<double>(prep_submit_stats_.count);
        fprintf(stderr,
                "[proxy-stats][w%u] polls/request=%.1f  "
                "(1.0 == every poll dispatched; high == worker spinning idle)\n",
                worker_idx, polls_per_req);
    }
}

void
ProxyWorker::publishCompletions(ChannelState &channel) {
    if (channel.error_latched) {
        return;
    }
    const bool stats_on = statsEnabled();
    while (!channel.inflight_requests.empty()) {
        ProxyRequestState &front = channel.inflight_requests.front();
        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backend_req_token);
            if (st == NIXL_IN_PROG) {
                break;
            }
        }
        // Inflight latency = backend completion observed - submit returned.
        // Skipped for short-circuit failures (front.status preset before submit).
        const auto t_complete = stats_on ? steady_clock::now() : steady_clock::time_point{};
        if (stats_on && front.submit_time.time_since_epoch().count() != 0) {
            inflight_stats_.record(ns_since(front.submit_time, t_complete));
        }
        NIXL_NVTX_MARK("prx:complete");
        NIXL_DEBUG << "ProxyWorker::publishCompletions: channel="
                   << channel.device_view.channel_id
                   << " op_idx=" << front.op_idx
                   << " status=" << st
                   << " token=" << front.backend_req_token;
        {
            NIXL_NVTX_RANGE("prx:publish");
            channel.completion_slot_host_->next_status = st;
            __atomic_store_n(&channel.completion_slot_host_->completed_idx,
                             front.op_idx, __ATOMIC_RELEASE);
        }
        if (stats_on) {
            const auto t_published = steady_clock::now();
            publish_stats_.record(ns_since(t_complete, t_published));
        }
        channel.inflight_requests.pop_front();
        if (st != NIXL_SUCCESS) {
            channel.error_latched = true;
            break;
        }
    }
}
