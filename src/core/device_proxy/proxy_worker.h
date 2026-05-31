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

#include <array>
#include <cstdint>
#include <limits>
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
                    uint64_t pthr_delay_us) noexcept;
        ~ProxyWorker();

        void start(uint32_t worker_idx);
        void join() noexcept;

        void
        runOnce();

    private:
        struct TimingStats {
            static constexpr size_t kHistBucketCount = 9;

            uint64_t count = 0;
            uint64_t sum_ns = 0;
            uint64_t min_ns = std::numeric_limits<uint64_t>::max();
            uint64_t max_ns = 0;
            double welford_mean_ns = 0.0;
            double welford_m2_ns = 0.0;
            std::vector<uint64_t> samples_ns;
            std::array<uint64_t, kHistBucketCount> hist_buckets{};

            void
            record(uint64_t duration_ns) noexcept;
        };

        bool
        tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission);

        void
        submitToBackend(nixlProxyChannelState &channel, const nixlProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(nixlProxyChannelState &channel);

        void
        printStats(uint32_t worker_idx) const noexcept;

        nixlDeviceProxyBackendAdapter *backend_ = nullptr;
        const nixlProxyMemViewRegistry *proxy_memview_registry_ = nullptr;
        uint32_t *shutdown_word_ = nullptr;
        nixlProxyChannelState *assigned_channels_ = nullptr;
        uint32_t assigned_channel_count_ = 0;
        uint64_t pthr_delay_us_ = 0;
        uint64_t run_once_iters_ = 0;
        uint64_t progress_calls_ = 0;
        uint64_t total_completion_polls_ = 0;
        TimingStats dequeue_stats_;
        TimingStats prepare_stats_;
        TimingStats submit_stats_;
        TimingStats post_submit_stats_;
        TimingStats progress_stats_;
        TimingStats check_completion_stats_;
        TimingStats post_submit_progress_stats_;
        TimingStats post_submit_check_stats_;
        TimingStats post_submit_wait_stats_;
        TimingStats publish_stats_;
        std::thread thread_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
