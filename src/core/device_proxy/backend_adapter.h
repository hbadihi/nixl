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

struct nixlBackendProxyXferDesc {
    nixl_mem_t mem_type = VRAM_SEG;
    nixlMetaDesc desc{};
};

struct nixlBackendProxySubmission {
    uint64_t op_idx = 0;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
    uint32_t channel_id = 0;
    uint64_t flags = 0;

    nixlBackendProxyXferDesc local{};
    nixlBackendProxyXferDesc remote{};

    size_t size = 0;
    uint64_t value = 0;
};

class nixlDeviceProxyBackendAdapter {
    public:
        virtual ~nixlDeviceProxyBackendAdapter() = default;

        virtual nixl_status_t
        init(uint32_t, uint32_t) {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        virtual nixl_status_t
        loadRemoteConnInfo(const std::string &, const nixl_blob_t &) {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Submit one proxy op. Returns NIXL_SUCCESS when the op completed at
        // submit time (request_token is 0 and must not be polled), NIXL_IN_PROG
        // with a nonzero request_token to poll via checkCompletion(), or a
        // terminal error. checkCompletion() releases the request on any
        // terminal status; a token abandoned before reaching a terminal status
        // must be handed to releaseRequest() instead.
        virtual nixl_status_t
        submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) = 0;

        virtual nixl_status_t
        checkCompletion(uint64_t request_token) = 0;

        virtual nixl_status_t
        releaseRequest(uint64_t) {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        virtual nixl_status_t
        progress() {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        virtual nixl_status_t
        shutdown() {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Debug-only (NIXL_EP_PROXY_STALL_LOG): per-UCX-worker submit histogram, e.g.
        // "w0=1234 w1=1230 ...", to verify QP utilization/spread across workers. Returns
        // empty when unsupported or when debug counting is disabled.
        virtual std::string
        workerSubmitHistogram() const {
            return {};
        }
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_BACKEND_ADAPTER_H
