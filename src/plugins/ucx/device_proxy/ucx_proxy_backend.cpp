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
#include "ucx_proxy_backend.h"
#include "../ucx_backend.h"
#include <cstddef>
#include <ucp/api/device/ucp_device_types.h>

namespace {
constexpr uint64_t kInvalidToken = 0;
}

nixl_status_t
nixlUcxProxyBackend::init(uint32_t worker_count, uint32_t channel_count) {
    if (engine_ == nullptr || worker_count == 0 || channel_count == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    worker_count_ = worker_count;
    channel_count_ = channel_count;
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
        stream_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::loadRemoteConnInfo(const std::string &remote_name,
                                        const nixl_blob_t &conn_info) {
    if (engine_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const nixl_status_t ret = engine_->loadRemoteConnInfo(remote_name, conn_info);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }
    return engine_->connect(remote_name);
}

nixl_status_t
nixlUcxProxyBackend::submit(const ResolvedProxySubmission &submission, uint64_t &request_token) {
    request_token = kInvalidToken;
    switch (submission.opcode) {
    case ProxyOpcode::PUT:
        return submitPut(submission, request_token);
    case ProxyOpcode::ATOMIC_ADD:
        return submitAtomicAdd(submission, request_token);
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

nixl_status_t
nixlUcxProxyBackend::checkCompletion(uint64_t request_token) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    const auto it = requests_.find(request_token);
    if (it == requests_.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    ProxyRequestState &request = it->second;
    if (!request.has_event) {
        const nixl_status_t st = request.status;
        requests_.erase(it);
        return st;
    }

    const cudaError_t err = cudaEventQuery(request.event);
    if (err == cudaSuccess) {
        cudaEventDestroy(request.event);
        const nixl_status_t st = request.status;
        requests_.erase(it);
        return st;
    }
    if (err == cudaErrorNotReady) {
        return NIXL_IN_PROG;
    }

    cudaEventDestroy(request.event);
    requests_.erase(it);
    return NIXL_ERR_BACKEND;
}

size_t
nixlUcxProxyBackend::progress() {
    // The fallback phase-1 implementation relies on CUDA stream progress.
    // Ask UCX to progress as well to keep transport state warm.
    return (engine_ != nullptr) ? static_cast<size_t>(engine_->progress()) : 0;
}

nixl_status_t
nixlUcxProxyBackend::shutdown() {
    std::lock_guard<std::mutex> lock(request_mutex_);
    for (auto &it : requests_) {
        if (it.second.has_event && it.second.event != nullptr) {
            cudaEventDestroy(it.second.event);
        }
    }
    requests_.clear();
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::getLocalAddress(nixlMemViewH memview,
                                     size_t index,
                                     size_t offset,
                                     void *&addr) const {
    addr = nullptr;
    if (memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ucp_device_local_mem_list_t header{};
    if (cudaMemcpy(&header, memview, sizeof(header), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    if (index >= header.length) {
        return NIXL_ERR_INVALID_PARAM;
    }

    uct_device_local_mem_list_elem_t element{};
    const auto *base = reinterpret_cast<const std::byte *>(memview);
    const auto *elem_ptr =
        base + offsetof(ucp_device_local_mem_list_t, mem_elements)
        + (index * sizeof(uct_device_local_mem_list_elem_t));
    if (cudaMemcpy(&element,
                   elem_ptr,
                   sizeof(element),
                   cudaMemcpyDeviceToHost)
        != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    addr = static_cast<void *>(static_cast<std::byte *>(element.addr) + offset);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::getRemoteAddress(nixlMemViewH memview,
                                      size_t index,
                                      size_t offset,
                                      void *&addr) const {
    addr = nullptr;
    if (memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ucp_device_remote_mem_list_t header{};
    if (cudaMemcpy(&header, memview, sizeof(header), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    if (index >= header.length) {
        return NIXL_ERR_INVALID_PARAM;
    }

    uct_device_remote_mem_list_elem_t element{};
    const auto *base = reinterpret_cast<const std::byte *>(memview);
    const auto *elem_ptr =
        base + offsetof(ucp_device_remote_mem_list_t, mem_elements)
        + (index * sizeof(uct_device_remote_mem_list_elem_t));
    if (cudaMemcpy(&element,
                   elem_ptr,
                   sizeof(element),
                   cudaMemcpyDeviceToHost)
        != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    addr = reinterpret_cast<void *>(element.addr + offset);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::submitPut(const ResolvedProxySubmission &submission, uint64_t &request_token) {
    if (stream_ == nullptr) {
        return NIXL_ERR_BACKEND;
    }

    void *src_addr = nullptr;
    void *dst_addr = nullptr;
    nixl_status_t status = getLocalAddress(submission.src_memview,
                                           submission.src_index,
                                           submission.src_offset,
                                           src_addr);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = getRemoteAddress(submission.dst_memview,
                              submission.dst_index,
                              submission.dst_offset,
                              dst_addr);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    if (cudaMemcpyAsync(dst_addr, src_addr, submission.size, cudaMemcpyDefault, stream_)
        != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    ProxyRequestState req{};
    req.has_event = true;
    req.status = NIXL_SUCCESS;
    if (cudaEventCreateWithFlags(&req.event, cudaEventDisableTiming) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    if (cudaEventRecord(req.event, stream_) != cudaSuccess) {
        cudaEventDestroy(req.event);
        req.event = nullptr;
        return NIXL_ERR_BACKEND;
    }

    request_token = makeRequestToken(std::move(req));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::submitAtomicAdd(const ResolvedProxySubmission &submission,
                                     uint64_t &request_token) {
    void *dst_addr = nullptr;
    const nixl_status_t status = getRemoteAddress(submission.dst_memview,
                                                  submission.dst_index,
                                                  submission.dst_offset,
                                                  dst_addr);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    uint64_t current = 0;
    if (cudaMemcpy(&current, dst_addr, sizeof(current), cudaMemcpyDefault) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    current += submission.value;
    if (cudaMemcpy(dst_addr, &current, sizeof(current), cudaMemcpyDefault) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    ProxyRequestState req{};
    req.has_event = false;
    req.status = NIXL_SUCCESS;
    request_token = makeRequestToken(std::move(req));
    return NIXL_SUCCESS;
}

uint64_t
nixlUcxProxyBackend::makeRequestToken(ProxyRequestState &&state) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    const uint64_t token = next_request_token_++;
    requests_.emplace(token, std::move(state));
    return token;
}
