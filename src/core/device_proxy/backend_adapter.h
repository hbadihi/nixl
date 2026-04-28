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
#include "backend_aux.h"
#include "proxy_protocol.h"

struct PreparedProxyTransferDesc {
    nixl_mem_t mem_type = DRAM_SEG;
    nixlMetaDesc desc{};
};

struct PreparedProxySubmission {
    uint64_t op_idx = 0;
    ProxyOpcode opcode = ProxyOpcode::PUT;
    uint32_t channel_id = 0;
    uint64_t flags = 0;

    PreparedProxyTransferDesc local{};
    PreparedProxyTransferDesc remote{};
    std::string remote_agent;

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
        submit(const PreparedProxySubmission &submission, uint64_t &request_token) = 0;

        virtual nixl_status_t
        checkCompletion(uint64_t request_token) = 0;

        virtual size_t
        progress() = 0;

        virtual nixl_status_t
        shutdown() = 0;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
