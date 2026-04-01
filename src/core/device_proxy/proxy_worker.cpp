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
#include <atomic>

// Shape-only handoff: worker dispatch is expected to resolve proxy memview IDs
// to backend memviews before submitting through the backend adapter.

ProxyWorker::ProxyWorker(DeviceProxyBackendAdapter *backend,
                         const ProxyMemViewRegistry *proxy_memview_registry,
                         std::atomic<uint32_t> *shutdown_word,
                         ChannelState *assigned_channels,
                         uint32_t assigned_channel_count) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      assigned_channels_(assigned_channels),
      assigned_channel_count_(assigned_channel_count) {}

void
ProxyWorker::runOnce() {
}

bool
ProxyWorker::tryDequeue(ChannelState &channel, ProxySubmission &submission) {
    (void)channel;
    (void)submission;
    return false;
}

nixl_status_t
ProxyWorker::dispatch(ChannelState &channel, const ProxySubmission &submission) {
    (void)channel;
    (void)submission;

    return NIXL_ERR_NOT_SUPPORTED;
}

void
ProxyWorker::driveBackendProgress() {
}

void
ProxyWorker::publishCompletions() {
}
