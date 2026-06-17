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

#include <atomic>
#include <cstdint>

#include "nixl_device_proxy.cuh"

namespace {

std::atomic<uint64_t> active_proxy_context_owner{0};

} // namespace

extern "C" cudaError_t
nixl_ep_proxy_publish_context(void *ctx, uint64_t owner_id) {
    if (ctx == nullptr || owner_id == 0) {
        return cudaErrorInvalidValue;
    }

    bool acquired_owner = false;
    uint64_t expected_owner = 0;
    if (active_proxy_context_owner.compare_exchange_strong(
            expected_owner, owner_id, std::memory_order_acq_rel)) {
        acquired_owner = true;
    } else if (expected_owner != owner_id) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status =
        nixlProxyPublishContext(static_cast<nixlProxyDeviceContextData *>(ctx));
    if (status != cudaSuccess && acquired_owner) {
        active_proxy_context_owner.store(0, std::memory_order_release);
    }
    return status;
}

extern "C" cudaError_t
nixl_ep_proxy_clear_context(uint64_t owner_id) {
    if (owner_id == 0) {
        return cudaErrorInvalidValue;
    }

    if (active_proxy_context_owner.load(std::memory_order_acquire) != owner_id) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = nixlProxyClearContext();
    if (status == cudaSuccess) {
        uint64_t expected_owner = owner_id;
        active_proxy_context_owner.compare_exchange_strong(
            expected_owner, 0, std::memory_order_acq_rel);
    }
    return status;
}
