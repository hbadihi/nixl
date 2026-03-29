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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
#define NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <nixl_types.h>
#include "proxy_protocol.h"

struct ResolvedProxySubmission {
    uint64_t op_idx = 0;
    ProxyOpcode opcode = ProxyOpcode::PUT;
    uint32_t channel_id = 0;
    uint64_t flags = 0;

    nixlMemViewH src_memview = nullptr;
    size_t src_index = 0;
    size_t src_offset = 0;

    nixlMemViewH dst_memview = nullptr;
    size_t dst_index = 0;
    size_t dst_offset = 0;

    size_t size = 0;
    uint64_t value = 0;
};

class DeviceProxyBackendAdapter {
    public:
        virtual ~DeviceProxyBackendAdapter() = default;

        virtual nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count) = 0;

        virtual nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info) = 0;

        virtual nixl_status_t
        submit(const ResolvedProxySubmission &submission, uint64_t &request_token) = 0;

        virtual nixl_status_t
        checkCompletion(uint64_t request_token) = 0;

        virtual size_t
        progress() = 0;

        virtual nixl_status_t
        shutdown() = 0;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
