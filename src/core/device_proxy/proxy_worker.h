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
        bool
        tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission);

        void
        submitToBackend(nixlProxyChannelState &channel, const nixlProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(nixlProxyChannelState &channel);

        // Drain-and-discard any stale ring entries + clear inflight/error/completion
        // state for a channel whose owning rank is (re)connecting. Runs only on the
        // worker thread (sole accessor of channel state), triggered by a generation bump.
        void
        resetChannel(nixlProxyChannelState &channel);

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
        std::thread thread_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
