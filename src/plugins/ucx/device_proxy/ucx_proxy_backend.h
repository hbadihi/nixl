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
#ifndef NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
#define NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "backend/backend_aux.h"
#include "../../../core/device_proxy/backend_adapter.h"

class nixlUcxEngine;

class nixlUcxProxyBackendAdapter : public nixlDeviceProxyBackendAdapter {
    public:
        explicit nixlUcxProxyBackendAdapter(nixlUcxEngine *engine = nullptr,
                                            bool progress_thread_enabled = false) noexcept
            : engine_(engine),
              progress_thread_enabled_(progress_thread_enabled),
              stall_log_enabled_(std::getenv("NIXL_EP_PROXY_STALL_LOG") != nullptr) {}

        ~nixlUcxProxyBackendAdapter() override = default;

        nixl_status_t
        submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) override;

        nixl_status_t
        checkCompletion(uint64_t request_token) override;

        nixl_status_t
        releaseRequest(uint64_t request_token) override;

        nixl_status_t
        progress() override;

        nixl_status_t
        shutdown() override;

        std::string
        workerSubmitHistogram() const override;

    private:
        // Deterministically map a proxy channel to a UCX worker so each channel uses
        // its own worker/EP/QP per peer. (The single proxy drain thread would otherwise
        // bind to one worker via getWorkerId()'s thread-local round-robin, collapsing
        // every channel onto a single QP.) channels_per_rank == num_workers, so for the
        // rank-encoded ring this recovers the lane.
        size_t
        workerIdForChannel(uint32_t channel_id) const;

        nixl_status_t
        submitPut(const nixlBackendProxySubmission &submission, uint64_t &request_token);

        nixl_status_t
        submitAtomicAdd(const nixlBackendProxySubmission &submission, uint64_t &request_token);

        nixlUcxEngine *engine_ = nullptr;
        bool progress_thread_enabled_ = false;

        // Debug-only QP-utilization counters (gated on NIXL_EP_PROXY_STALL_LOG). Indexed by
        // worker_id = channel_id % num_workers; a fixed cap avoids needing an init() hook.
        // Submits with worker_id >= cap are not counted (num_workers never approaches this).
        static constexpr size_t kMaxTrackedWorkers = 256;
        bool stall_log_enabled_ = false;
        std::array<std::atomic<uint64_t>, kMaxTrackedWorkers> worker_submit_counts_{};
};

#endif // NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
