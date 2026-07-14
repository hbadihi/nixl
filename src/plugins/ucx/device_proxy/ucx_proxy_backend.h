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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "backend/backend_aux.h"
#include "../../../core/device_proxy/backend_adapter.h"

class nixlUcxEngine;

struct nixlUcxProxyRankMapping {
    uint32_t local_rank;
    uint32_t channels_per_rank;
};

std::optional<size_t>
nixlUcxProxyWorkerIdForChannel(uint32_t channel_id,
                               size_t num_workers,
                               uint32_t channel_count,
                               const nixlUcxProxyRankMapping &mapping) noexcept;

class nixlUcxProxyBackendAdapter : public nixlDeviceProxyBackendAdapter {
    public:
        explicit nixlUcxProxyBackendAdapter(nixlUcxEngine *engine = nullptr,
                                            bool progress_thread_enabled = false,
                                            std::optional<nixlUcxProxyRankMapping> rank_mapping =
                                                std::nullopt) noexcept
            : engine_(engine),
              progress_thread_enabled_(progress_thread_enabled),
              rank_mapping_(rank_mapping) {}

        ~nixlUcxProxyBackendAdapter() override = default;

        nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count) override;

        nixl_status_t
        submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) override;

        nixl_status_t
        checkCompletion(uint64_t request_token) override;

        nixl_status_t
        releaseRequest(uint64_t request_token) override;

        nixl_status_t
        progress() override;

        nixl_status_t
        progress(uint32_t channel_id) override;

        nixl_status_t
        shutdown() override;

    private:
        // Rank-encoded ring IDs include an unused local-rank band. Compact that hole
        // before selecting a UCX worker so each active (destination, lane) ring owns a
        // distinct worker. Proxy users without rank mapping retain the legacy modulo map.
        std::optional<size_t>
        workerIdForChannel(uint32_t channel_id) const;

        nixl_status_t
        submitPut(const nixlBackendProxySubmission &submission, uint64_t &request_token);

        nixl_status_t
        submitAtomicAdd(const nixlBackendProxySubmission &submission, uint64_t &request_token);

        nixlUcxEngine *engine_ = nullptr;
        bool progress_thread_enabled_ = false;
        std::optional<nixlUcxProxyRankMapping> rank_mapping_;
        uint32_t channel_count_ = 0;
};

#endif // NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
