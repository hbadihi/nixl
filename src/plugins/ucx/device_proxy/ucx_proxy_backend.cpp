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
#include "nixl_log.h"

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
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::loadRemoteConnInfo(const std::string &remote_name,
                                        const nixl_blob_t &conn_info) {
    if (engine_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (engine_->checkConn(remote_name) != NIXL_SUCCESS) {
        const nixl_status_t ret = engine_->loadRemoteConnInfo(remote_name, conn_info);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
    }
    return engine_->connect(remote_name);
}

void
nixlUcxProxyBackend::storeLocalMeta(nixlMemViewH mvh, const nixl_meta_dlist_t &dlist) {
    MemViewMeta meta;
    meta.is_remote = false;
    meta.mem_type = dlist.getType();

    const size_t count = dlist.descCount();
    meta.entries.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto &desc = dlist[i];
        auto *priv = static_cast<nixlUcxPrivateMetadata *>(desc.metadataP);
        StoredEntry entry;
        entry.base_addr = reinterpret_cast<uintptr_t>(priv->getMem().getBase());
        entry.metadataP = desc.metadataP;
        meta.entries.push_back(entry);
    }

    NIXL_DEBUG << "nixlUcxProxyBackend::storeLocalMeta: mvh=" << mvh
               << " entries=" << count << " mem_type=" << static_cast<int>(meta.mem_type);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_store_[mvh] = std::move(meta);
}

void
nixlUcxProxyBackend::storeRemoteMeta(nixlMemViewH mvh, const nixl_remote_meta_dlist_t &dlist) {
    MemViewMeta meta;
    meta.is_remote = true;
    meta.mem_type = dlist.getType();

    const size_t count = dlist.descCount();
    for (size_t i = 0; i < count; ++i) {
        const auto &desc = dlist[i];
        if (meta.remote_agent.empty() && desc.remoteAgent != nixl_null_agent) {
            meta.remote_agent = desc.remoteAgent;
        }
        StoredEntry entry;
        entry.base_addr = desc.addr;
        entry.metadataP = desc.metadataP;
        meta.entries.push_back(entry);
    }

    NIXL_DEBUG << "nixlUcxProxyBackend::storeRemoteMeta: mvh=" << mvh
               << " entries=" << count
               << " remote_agent='" << meta.remote_agent << "'"
               << " mem_type=" << static_cast<int>(meta.mem_type);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_store_[mvh] = std::move(meta);
}

void
nixlUcxProxyBackend::clearMeta(nixlMemViewH mvh) {
    NIXL_DEBUG << "nixlUcxProxyBackend::clearMeta: mvh=" << mvh;
    std::lock_guard<std::mutex> lock(meta_mutex_);
    meta_store_.erase(mvh);
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
nixlUcxProxyBackend::submitPut(const ResolvedProxySubmission &submission, uint64_t &request_token) {
    MemViewMeta src_meta;
    MemViewMeta dst_meta;

    {
        std::lock_guard<std::mutex> lock(meta_mutex_);
        auto src_it = meta_store_.find(submission.src_memview);
        auto dst_it = meta_store_.find(submission.dst_memview);
        if (src_it == meta_store_.end() || dst_it == meta_store_.end()) {
            NIXL_DEBUG << "nixlUcxProxyBackend::submitPut: metadata not found"
                       << " src_mvh=" << submission.src_memview
                       << " dst_mvh=" << submission.dst_memview;
            return NIXL_ERR_NOT_FOUND;
        }
        src_meta = src_it->second;
        dst_meta = dst_it->second;
    }

    if (submission.src_index >= src_meta.entries.size() ||
        submission.dst_index >= dst_meta.entries.size()) {
        NIXL_DEBUG << "nixlUcxProxyBackend::submitPut: index out of range"
                   << " src_index=" << submission.src_index
                   << " src_count=" << src_meta.entries.size()
                   << " dst_index=" << submission.dst_index
                   << " dst_count=" << dst_meta.entries.size();
        return NIXL_ERR_INVALID_PARAM;
    }

    const StoredEntry &src_entry = src_meta.entries[submission.src_index];
    const StoredEntry &dst_entry = dst_meta.entries[submission.dst_index];

    nixl_meta_dlist_t local_list(src_meta.mem_type);
    local_list.addDesc(nixlMetaDesc(
        src_entry.base_addr + submission.src_offset,
        submission.size,
        0,
        src_entry.metadataP));

    nixl_meta_dlist_t remote_list(dst_meta.mem_type);
    remote_list.addDesc(nixlMetaDesc(
        dst_entry.base_addr + submission.dst_offset,
        submission.size,
        0,
        dst_entry.metadataP));

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(
        NIXL_WRITE, local_list, remote_list, dst_meta.remote_agent, handle);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackend::submitPut: prepXfer failed status=" << status;
        return status;
    }

    status = engine_->postXfer(
        NIXL_WRITE, local_list, remote_list, dst_meta.remote_agent, handle);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackend::submitPut: postXfer failed status=" << status;
        engine_->releaseReqH(handle);
        return status;
    }

    request_token = trackRequest(handle);
    NIXL_DEBUG << "nixlUcxProxyBackend::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex
               << (src_entry.base_addr + submission.src_offset) << std::dec
               << " dst_addr=0x" << std::hex
               << (dst_entry.base_addr + submission.dst_offset) << std::dec
               << " size=" << submission.size
               << " remote_agent='" << dst_meta.remote_agent << "'"
               << " token=" << request_token;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackend::submitAtomicAdd(const ResolvedProxySubmission &,
                                     uint64_t &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlUcxProxyBackend::checkCompletion(uint64_t request_token) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    const auto it = tracked_requests_.find(request_token);
    if (it == tracked_requests_.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlBackendReqH *handle = it->second;
    const nixl_status_t status = engine_->checkXfer(handle);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "nixlUcxProxyBackend::checkCompletion: token=" << request_token
               << " status=" << status;
    engine_->releaseReqH(handle);
    tracked_requests_.erase(it);
    return status;
}

size_t
nixlUcxProxyBackend::progress() {
    // TODO: progress each UCX worker.
    return (engine_ != nullptr) ? static_cast<size_t>(engine_->progress()) : 0;
}

nixl_status_t
nixlUcxProxyBackend::shutdown() {
    NIXL_INFO << "nixlUcxProxyBackend::shutdown: releasing "
              << tracked_requests_.size() << " tracked request(s) and "
              << meta_store_.size() << " stored metadata entries";
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        for (auto &[token, handle] : tracked_requests_) {
            engine_->releaseReqH(handle);
        }
        tracked_requests_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(meta_mutex_);
        meta_store_.clear();
    }
    return NIXL_SUCCESS;
}

uint64_t
nixlUcxProxyBackend::trackRequest(nixlBackendReqH *handle) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    const uint64_t token = next_request_token_++;
    tracked_requests_.emplace(token, handle);
    return token;
}
