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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

enum class ProxyOpcode : uint32_t {
    PUT = 0,
    ATOMIC_ADD = 1,
};

struct ProxySubmission {
    uint64_t op_idx = 0;
    ProxyOpcode opcode = ProxyOpcode::PUT;
    uint32_t channel_id = 0;
    uint64_t flags = 0;

    uint64_t src_proxy_memview_id = 0;
    size_t src_index = 0;
    size_t src_offset = 0;

    uint64_t dst_proxy_memview_id = 0;
    size_t dst_index = 0;
    size_t dst_offset = 0;

    size_t size = 0;
    uint64_t value = 0;
};

struct WorkRing {
    ProxySubmission *records = nullptr;
    uint32_t *producer_idx = nullptr;
    uint32_t *consumer_idx = nullptr;
    uint32_t depth = 0;
};

struct CompletionSlot {
    uint64_t completed_idx = 0;
    nixl_status_t next_status = NIXL_IN_PROG;
};

struct ProxyChannelView {
    WorkRing *work_ring = nullptr;
    CompletionSlot *completion_slot = nullptr;
    uint32_t channel_id = 0;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_PROTOCOL_H
