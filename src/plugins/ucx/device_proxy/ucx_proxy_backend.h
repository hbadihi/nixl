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

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

#include "../../../core/device_proxy/backend_adapter.h"

class nixlUcxEngine;

class nixlUcxProxyBackend : public DeviceProxyBackendAdapter {
    public:
        explicit nixlUcxProxyBackend(nixlUcxEngine *engine = nullptr) noexcept
            : engine_(engine) {}

        ~nixlUcxProxyBackend() override = default;

        nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count) override;

        nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info) override;

        nixl_status_t
        submit(const ResolvedProxySubmission &submission, uint64_t &request_token) override;

        nixl_status_t
        checkCompletion(uint64_t request_token) override;

        size_t
        progress() override;

        nixl_status_t
        shutdown() override;

    private:
        struct ProxyRequestState {
            bool has_event = false;
            cudaEvent_t event = nullptr;
            nixl_status_t status = NIXL_IN_PROG;
        };

        nixl_status_t
        getLocalAddress(nixlMemViewH memview, size_t index, size_t offset, void *&addr) const;

        nixl_status_t
        getRemoteAddress(nixlMemViewH memview, size_t index, size_t offset, void *&addr) const;

        nixl_status_t
        submitPut(const ResolvedProxySubmission &submission, uint64_t &request_token);

        nixl_status_t
        submitAtomicAdd(const ResolvedProxySubmission &submission, uint64_t &request_token);

        uint64_t
        makeRequestToken(ProxyRequestState &&state);

        nixlUcxEngine *engine_ = nullptr;
        uint32_t worker_count_ = 0;
        uint32_t channel_count_ = 0;
        cudaStream_t stream_ = nullptr;
        std::mutex request_mutex_;
        std::unordered_map<uint64_t, ProxyRequestState> requests_;
        uint64_t next_request_token_ = 1;
};

#endif // NIXL_SRC_PLUGINS_UCX_DEVICE_PROXY_UCX_PROXY_BACKEND_H
