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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
#include "proxy_protocol.h"

class nixlDeviceProxyBackendAdapter;
class nixlProxyMemViewRegistry;
struct nixlProxyChannelState;

class ProxyWorker {
    public:
        ProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                    const nixlProxyMemViewRegistry *proxy_memview_registry,
                    uint32_t *shutdown_word,
                    nixlProxyChannelState *assigned_channels,
                    uint32_t assigned_channel_count,
                    uint64_t pthr_delay_us,
                    std::atomic<uint64_t> *submitted_work_count,
                    std::atomic<uint64_t> *assigned_generations = nullptr) noexcept;
        ~ProxyWorker();

        void start(uint32_t worker_idx);
        void join() noexcept;

        void
        runOnce();

    private:
        struct ChannelDebugCounters {
            uint64_t dequeued_put = 0;
            uint64_t dequeued_atomic = 0;
            uint64_t completed_put = 0;
            uint64_t completed_atomic = 0;
            uint64_t prepare_errors = 0;
            uint64_t submit_errors = 0;
        };

        bool
        tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission);

        void
        submitToBackend(nixlProxyChannelState &channel, const nixlProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(nixlProxyChannelState &channel);

        // Fire-and-forget reap (NIXL_EP_PROXY_FIRE_AND_FORGET): poll in-flight tokens and free
        // terminal backend requests out-of-order, without publishing completed_idx, preserving
        // FIFO order, or error-latching. For EP, which never reads completions
        // (no nixlGetGpuXferStatus), this removes the head-of-line block and permanent latch
        // that can otherwise wedge a channel.
        void
        reapCompletions(nixlProxyChannelState &channel);

        // Drain-and-discard any stale ring entries + clear inflight/error/completion
        // state for a channel whose owning rank is (re)connecting. Runs only on the
        // worker thread (sole accessor of channel state), triggered by a generation bump.
        void
        resetChannel(nixlProxyChannelState &channel);

        // Diagnostics: rate-limited worker/ring/inflight dump when
        // NIXL_EP_PROXY_STALL_LOG is set.
        void
        maybeLogStalls();

        nixlDeviceProxyBackendAdapter *backend_ = nullptr;
        const nixlProxyMemViewRegistry *proxy_memview_registry_ = nullptr;
        uint32_t *shutdown_word_ = nullptr;
        nixlProxyChannelState *assigned_channels_ = nullptr;
        uint32_t assigned_channel_count_ = 0;
        uint64_t pthr_delay_us_ = 0;
        std::atomic<uint64_t> *submitted_work_count_ = nullptr;
        // Per-assigned-channel generation slice (runtime-owned; nullptr if disabled). The
        // runtime bumps a band's generation on a remote-agent change; this worker compares
        // against last_seen_gen_ in runOnce and reconciles (resetChannel) when they differ.
        std::atomic<uint64_t> *assigned_generations_ = nullptr;
        std::vector<uint64_t> last_seen_gen_;

        // Diagnostics (NIXL_EP_PROXY_STALL_LOG): periodically log a worker heartbeat and
        // any assigned channel with outstanding inflight requests or published ring records.
        // Read-only; no behavior change. The heartbeat distinguishes an idle worker from one
        // stuck in backend progress; scanning the ring detects published records stranded
        // behind an unpublished head slot.
        bool stall_log_enabled_ = false;
        // Fire-and-forget completion handling (NIXL_EP_PROXY_FIRE_AND_FORGET): skip the
        // FIFO/latch/device-publish path in favor of reapCompletions().
        bool fire_and_forget_ = false;
        // This worker's index; only worker 0 dumps the process-wide QP histogram so the
        // (backend-global) counters aren't logged once per drain thread.
        uint32_t worker_idx_ = 0;
        std::chrono::steady_clock::time_point last_stall_log_{};
        uint64_t run_once_count_ = 0;
        uint64_t last_logged_run_once_count_ = 0;
        std::vector<ChannelDebugCounters> channel_debug_counters_;

        std::thread thread_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
