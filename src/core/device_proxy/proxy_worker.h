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

#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>
#include "proxy_protocol.h"

class DeviceProxyBackendAdapter;
class ProxyMemViewRegistry;
struct ChannelState;

/** Per-stage running stats for the proxy worker.  All times in nanoseconds.
 *  Cheap to update (no atomics — single-writer thread) and dumped once at
 *  thread exit.  Set NIXL_PROXY_STATS=0 to disable the per-thread summary. */
struct ProxyWorkerStats {
    uint64_t count   = 0;
    uint64_t sum_ns  = 0;
    uint64_t min_ns  = std::numeric_limits<uint64_t>::max();
    uint64_t max_ns  = 0;

    void
    record(uint64_t ns) noexcept {
        ++count;
        sum_ns += ns;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
    }
};

class ProxyWorker {
    public:
        ProxyWorker(DeviceProxyBackendAdapter *backend,
                    const ProxyMemViewRegistry *proxy_memview_registry,
                    uint32_t *shutdown_word,
                    ChannelState *assigned_channels,
                    uint32_t assigned_channel_count) noexcept;
        ~ProxyWorker();

        void start(uint32_t worker_idx);
        void join() noexcept;

        void
        runOnce();

    private:
        bool
        tryDequeue(ChannelState &channel, ProxySubmission &submission);

        nixl_status_t
        submitToBackend(ChannelState &channel, const ProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(ChannelState &channel);

        void
        logStatsSummary(uint32_t worker_idx) const;

        DeviceProxyBackendAdapter *backend_ = nullptr;
        const ProxyMemViewRegistry *proxy_memview_registry_ = nullptr;
        uint32_t *shutdown_word_ = nullptr;
        ChannelState *assigned_channels_ = nullptr;
        uint32_t assigned_channel_count_ = 0;
        std::thread thread_;

        // Per-stage timing.  Touched only by the worker thread.
        ProxyWorkerStats prep_submit_stats_;     // tryDequeue→submit return
        ProxyWorkerStats inflight_stats_;        // submit return→completion seen
        ProxyWorkerStats publish_stats_;         // completion seen→completion published
        uint64_t         run_once_count_   = 0;  // total runOnce iterations
        uint64_t         progress_count_   = 0;  // total backend->progress calls
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
