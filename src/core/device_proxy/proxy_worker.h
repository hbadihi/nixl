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

#include <cstdint>
#include <thread>
#include "proxy_protocol.h"

class DeviceProxyBackendAdapter;
class ProxyMemViewRegistry;
struct ChannelState;

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
        dispatch(ChannelState &channel, const ProxySubmission &submission);

        void
        driveBackendProgress();

        void
        publishCompletions(ChannelState &channel);

        DeviceProxyBackendAdapter *backend_ = nullptr;
        const ProxyMemViewRegistry *proxy_memview_registry_ = nullptr;
        uint32_t *shutdown_word_ = nullptr;
        ChannelState *assigned_channels_ = nullptr;
        uint32_t assigned_channel_count_ = 0;
        std::thread thread_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_WORKER_H
