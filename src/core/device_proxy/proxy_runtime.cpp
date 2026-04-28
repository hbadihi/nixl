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
#include "proxy_runtime.h"
#include "backend_adapter.h"
#include "nixl_types.h"
#include "proxy_worker.h"
#include "nixl_log.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

nixl_status_t
ProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                           nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    RegistryEntry entry;
    entry.proxy_memview_id = next_proxy_memview_id_;
    entry.proxy_memview = reinterpret_cast<nixlMemViewH>(entry.proxy_memview_id);
    entry.backend_memview = backend_memview;
    entries_.push_back(entry);

    *proxy_memview = entry.proxy_memview;
    ++next_proxy_memview_id_;
    NIXL_DEBUG << "ProxyMemViewRegistry::register: backend_mvh="
               << backend_memview << " -> proxy_id="
               << (next_proxy_memview_id_ - 1);
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyMemViewRegistry::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    entry->state = EntryState::Retired;
    NIXL_DEBUG << "ProxyMemViewRegistry::unregister: proxy_id="
               << entry->proxy_memview_id;
    return NIXL_SUCCESS;
}

bool
ProxyMemViewRegistry::resolveProxyMemView(nixlMemViewH proxy_memview,
                                          nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == EntryState::Retired) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

bool
ProxyMemViewRegistry::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                            nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForId(proxy_memview_id);
    if (entry == nullptr || entry->state == EntryState::Retired) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

nixl_status_t
ProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                    const nixl_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == EntryState::Retired) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillLocalMetadata(dlist, entry->local_metadata);
    entry->remote_metadata = RemoteMetadata{};
    entry->metadata_kind = MetadataKind::Local;
    entry->state = EntryState::Ready;

    NIXL_DEBUG << "ProxyMemViewRegistry::storeMetadata(local): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                    const nixl_remote_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == EntryState::Retired) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillRemoteMetadata(dlist, entry->remote_metadata);
    entry->local_metadata = LocalMetadata{};
    entry->metadata_kind = MetadataKind::Remote;
    entry->state = EntryState::Ready;

    NIXL_DEBUG << "ProxyMemViewRegistry::storeMetadata(remote): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyMemViewRegistry::prepareSubmission(const ProxySubmission &submission,
                                        PreparedProxySubmission &prepared_submission) const {
    const RegistryEntry *dst_entry = getEntryForId(submission.dst_proxy_memview_id);
    if (dst_entry == nullptr || dst_entry->state != EntryState::Ready) {
        NIXL_DEBUG << "ProxyMemViewRegistry::prepareSubmission: dst not ready"
                   << " dst_proxy_id=" << submission.dst_proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (dst_entry->metadata_kind != MetadataKind::Remote) {
        NIXL_DEBUG << "ProxyMemViewRegistry::prepareSubmission: dst metadata kind invalid"
                   << " dst_proxy_id=" << submission.dst_proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (submission.dst_index >= dst_entry->remote_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    prepared_submission = PreparedProxySubmission{};
    prepared_submission.op_idx = submission.op_idx;
    prepared_submission.opcode = submission.opcode;
    prepared_submission.channel_id = submission.channel_id;
    prepared_submission.flags = submission.flags;
    prepared_submission.size = submission.size;
    prepared_submission.value = submission.value;
    prepared_submission.remote_agent = dst_entry->remote_metadata.remote_agent;
    prepared_submission.remote.mem_type = dst_entry->remote_metadata.mem_type;
    prepared_submission.remote.desc = nixlMetaDesc(
        dst_entry->remote_metadata.entries[submission.dst_index].base_addr + submission.dst_offset,
        submission.size,
        0,
        dst_entry->remote_metadata.entries[submission.dst_index].metadata);

    if (submission.opcode == ProxyOpcode::PUT) {
        const RegistryEntry *src_entry = getEntryForId(submission.src_proxy_memview_id);
        if (src_entry == nullptr || src_entry->state != EntryState::Ready) {
            NIXL_DEBUG << "ProxyMemViewRegistry::prepareSubmission: src not ready"
                       << " src_proxy_id=" << submission.src_proxy_memview_id;
            return NIXL_ERR_NOT_FOUND;
        }
        if (src_entry->metadata_kind != MetadataKind::Local) {
            NIXL_DEBUG << "ProxyMemViewRegistry::prepareSubmission: src metadata kind invalid"
                       << " src_proxy_id=" << submission.src_proxy_memview_id;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (submission.src_index >= src_entry->local_metadata.entries.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        prepared_submission.local.mem_type = src_entry->local_metadata.mem_type;
        prepared_submission.local.desc = nixlMetaDesc(
            src_entry->local_metadata.entries[submission.src_index].base_addr + submission.src_offset,
            submission.size,
            0,
            src_entry->local_metadata.entries[submission.src_index].metadata);
    }

    return NIXL_SUCCESS;
}

void
ProxyMemViewRegistry::clear() noexcept {
    for (auto &entry : entries_) {
        entry.state = EntryState::Retired;
    }
}

ProxyMemViewRegistry::RegistryEntry *
ProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) {
    return getEntryForId(reinterpret_cast<uint64_t>(proxy_memview));
}

const ProxyMemViewRegistry::RegistryEntry *
ProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) const {
    return getEntryForId(reinterpret_cast<uint64_t>(proxy_memview));
}

ProxyMemViewRegistry::RegistryEntry *
ProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) {
    if (proxy_memview_id < 1
        || proxy_memview_id >= next_proxy_memview_id_
        || proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

const ProxyMemViewRegistry::RegistryEntry *
ProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) const {
    if (proxy_memview_id < 1
        || proxy_memview_id >= next_proxy_memview_id_
        || proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

void
ProxyMemViewRegistry::fillLocalMetadata(const nixl_meta_dlist_t &dlist,
                                        LocalMetadata &out) {
    out = LocalMetadata{};
    out.mem_type = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(StoredEntry{desc.addr, desc.metadataP});
    }
}

void
ProxyMemViewRegistry::fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist,
                                         RemoteMetadata &out) {
    out = RemoteMetadata{};
    out.mem_type = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        if (out.remote_agent.empty() && desc.remoteAgent != nixl_null_agent) {
            out.remote_agent = desc.remoteAgent;
        }
        out.entries.push_back(StoredEntry{desc.addr, desc.metadataP});
    }
}

nixl_status_t
ChannelState::allocate(uint32_t channel_id, uint32_t depth) {
    NIXL_INFO << "ChannelState::allocate: channel_id=" << channel_id
              << " depth=" << depth;
    if (cudaMallocHost(&work_ring_,       sizeof(WorkRing))                != cudaSuccess
     || cudaMallocHost(&records_,         sizeof(ProxySubmission) * depth) != cudaSuccess
     || cudaMallocHost(reinterpret_cast<void **>(&consumer_idx_host_),
                       sizeof(uint32_t))                                 != cudaSuccess
     || cudaMallocHost(reinterpret_cast<void **>(&producer_idx_host_),
                       sizeof(uint32_t))                                 != cudaSuccess
     || cudaMallocHost(&completion_slot_host_, sizeof(CompletionSlot))     != cudaSuccess) {
        NIXL_ERROR << "ChannelState::allocate: CUDA allocation failed for channel "
                   << channel_id;
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    void *consumer_dev = nullptr;
    if (cudaHostGetDevicePointer(&consumer_dev, consumer_idx_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    consumer_idx_dev_ = static_cast<uint32_t *>(consumer_dev);

    void *producer_dev = nullptr;
    if (cudaHostGetDevicePointer(&producer_dev, producer_idx_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    producer_idx_dev_ = static_cast<uint32_t *>(producer_dev);

    void *completion_dev = nullptr;
    if (cudaHostGetDevicePointer(&completion_dev, completion_slot_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    completion_slot_dev_ = static_cast<CompletionSlot *>(completion_dev);

    for (uint32_t i = 0; i < depth; ++i) {
        records_[i] = ProxySubmission{};
    }
    __atomic_store_n(producer_idx_host_, 0, __ATOMIC_RELEASE);
    __atomic_store_n(consumer_idx_host_, 0, __ATOMIC_RELEASE);
    completion_slot_host_->next_status = NIXL_IN_PROG;
    __atomic_store_n(&completion_slot_host_->completed_idx,
                     uint64_t{0}, __ATOMIC_RELEASE);
    *work_ring_ = WorkRing{
        records_,
        producer_idx_dev_,
        consumer_idx_dev_,
        depth,
    };
    device_view       = ProxyChannelView{ work_ring_, completion_slot_dev_, channel_id };

    inflight_requests.clear();
    NIXL_INFO << "ChannelState::allocate: channel " << channel_id << " ready"
              << " work_ring=" << work_ring_
              << " records=" << records_
              << " producer_idx(host)=" << producer_idx_host_
              << " producer_idx(dev)=" << producer_idx_dev_
              << " consumer_idx(host)=" << consumer_idx_host_
              << " consumer_idx(dev)=" << consumer_idx_dev_
              << " completion_slot(host)=" << completion_slot_host_
              << " completion_slot(dev)=" << completion_slot_dev_;
    return NIXL_SUCCESS;
}

void
ChannelState::deallocate() noexcept {
    if (completion_slot_host_) {
        cudaFreeHost(completion_slot_host_);
        completion_slot_host_ = nullptr;
        completion_slot_dev_  = nullptr;
    }
    if (producer_idx_host_) {
        cudaFreeHost(producer_idx_host_);
        producer_idx_host_ = nullptr;
        producer_idx_dev_  = nullptr;
    }
    if (consumer_idx_host_) {
        cudaFreeHost(consumer_idx_host_);
        consumer_idx_host_ = nullptr;
        consumer_idx_dev_  = nullptr;
    }
    if (records_)         { cudaFreeHost(records_);         records_         = nullptr; }
    if (work_ring_)       { cudaFreeHost(work_ring_);       work_ring_       = nullptr; }
    device_view = ProxyChannelView{};
}

ChannelState::~ChannelState() {
    deallocate();
}

ChannelState::ChannelState(ChannelState &&other) noexcept
    : device_view(other.device_view),
      inflight_requests(std::move(other.inflight_requests)),
      work_ring_(other.work_ring_),
      records_(other.records_),
      producer_idx_host_(other.producer_idx_host_),
      producer_idx_dev_(other.producer_idx_dev_),
      consumer_idx_host_(other.consumer_idx_host_),
      consumer_idx_dev_(other.consumer_idx_dev_),
      completion_slot_host_(other.completion_slot_host_),
      completion_slot_dev_(other.completion_slot_dev_) {
    other.work_ring_            = nullptr;
    other.records_              = nullptr;
    other.producer_idx_host_    = nullptr;
    other.producer_idx_dev_     = nullptr;
    other.consumer_idx_host_    = nullptr;
    other.consumer_idx_dev_     = nullptr;
    other.completion_slot_host_ = nullptr;
    other.completion_slot_dev_  = nullptr;
    other.device_view      = ProxyChannelView{};
}

ChannelState &
ChannelState::operator=(ChannelState &&other) noexcept {
    if (this != &other) {
        deallocate();
        device_view       = other.device_view;
        inflight_requests = std::move(other.inflight_requests);
        work_ring_           = other.work_ring_;
        records_             = other.records_;
        producer_idx_host_   = other.producer_idx_host_;
        producer_idx_dev_    = other.producer_idx_dev_;
        consumer_idx_host_   = other.consumer_idx_host_;
        consumer_idx_dev_    = other.consumer_idx_dev_;
        completion_slot_host_    = other.completion_slot_host_;
        completion_slot_dev_     = other.completion_slot_dev_;
        other.work_ring_            = nullptr;
        other.records_              = nullptr;
        other.producer_idx_host_    = nullptr;
        other.producer_idx_dev_     = nullptr;
        other.consumer_idx_host_    = nullptr;
        other.consumer_idx_dev_     = nullptr;
        other.completion_slot_host_ = nullptr;
        other.completion_slot_dev_  = nullptr;
        other.device_view      = ProxyChannelView{};
    }
    return *this;
}

ProxyRuntime::ProxyRuntime() = default;

ProxyRuntime::~ProxyRuntime() {
    if (backend_) {
        shutdown();
    }
}

nixl_status_t
ProxyRuntime::init(DeviceProxyBackendAdapter *backend,
                   uint32_t channel_count,
                   uint32_t worker_count) {
    NIXL_INFO << "ProxyRuntime::init: channel_count=" << channel_count
              << " worker_count=" << worker_count
              << " backend=" << backend;
    if (backend == nullptr || channel_count == 0 || worker_count == 0) {
        NIXL_ERROR << "ProxyRuntime::init: invalid params";
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = backend;
    memview_registry_.clear();

    if (cudaMallocHost(reinterpret_cast<void **>(&shutdown_word_host_),
                       sizeof(uint32_t)) != cudaSuccess) {
        NIXL_ERROR << "ProxyRuntime::init: failed to allocate shutdown_word";
        shutdown_word_host_ = nullptr;
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    void *shutdown_dev = nullptr;
    if (cudaHostGetDevicePointer(&shutdown_dev, shutdown_word_host_, 0) != cudaSuccess) {
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    shutdown_word_dev_ = static_cast<uint32_t *>(shutdown_dev);
    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(ProxyControlState::Running),
                     __ATOMIC_RELEASE);

    worker_count = std::min(worker_count, channel_count);
    NIXL_INFO << "ProxyRuntime::init: effective worker_count=" << worker_count
              << " (clamped to channel_count)";

    nixl_status_t rc = backend_->init(worker_count, channel_count);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::init: backend init failed: " << rc;
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_  = nullptr;
        backend_ = nullptr;
        return rc;
    }

    channels_.resize(channel_count);
    for (uint32_t channel_id = 0; channel_id < channel_count; ++channel_id) {
        rc = channels_[channel_id].allocate(channel_id, ring_depth_);
        if (rc != NIXL_SUCCESS) {
            channels_.clear();
            backend_->shutdown();
            cudaFreeHost(shutdown_word_host_);
            shutdown_word_host_ = nullptr;
            shutdown_word_dev_  = nullptr;
            backend_ = nullptr;
            return rc;
        }
    }

    if (cudaMallocHost(&device_channel_views_,
                       sizeof(ProxyChannelView) * channel_count) != cudaSuccess) {
        channels_.clear();
        backend_->shutdown();
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_  = nullptr;
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    for (uint32_t channel_id = 0; channel_id < channel_count; ++channel_id) {
        device_channel_views_[channel_id] = channels_[channel_id].device_view;
    }

    if (cudaMallocHost(&device_context_,
                       sizeof(ProxyDeviceContextData)) != cudaSuccess) {
        cudaFreeHost(device_channel_views_);
        device_channel_views_ = nullptr;
        channels_.clear();
        backend_->shutdown();
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_  = nullptr;
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    *device_context_ = ProxyDeviceContextData{
        device_channel_views_,
        channel_count,
        shutdown_word_dev_
    };

    workers_.clear();
    workers_.reserve(worker_count);

    for (uint32_t w = 0; w < worker_count; ++w) {
        uint32_t first_ch = (w * channel_count) / worker_count;
        uint32_t end_ch   = ((w + 1) * channel_count) / worker_count;
        uint32_t n_ch     = end_ch - first_ch;

        NIXL_INFO << "ProxyRuntime::init: worker " << w
                  << " assigned channels [" << first_ch << ", " << end_ch << ")";
        workers_.push_back(std::make_unique<ProxyWorker>(
            backend_,
            &memview_registry_,
            shutdown_word_host_,
            &channels_[first_ch],
            n_ch));
    }

    NIXL_INFO << "ProxyRuntime::init: complete — "
              << channel_count << " channels, "
              << worker_count << " workers, "
              << "device_context=" << device_context_;
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyRuntime::loadRemoteConnInfo(const std::string &remote_name,
                                 const nixl_blob_t &conn_info) {
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: remote='" << remote_name
              << "' conn_info_size=" << conn_info.size();
    if (backend_ == nullptr) {
        NIXL_ERROR << "ProxyRuntime::loadRemoteConnInfo: no backend";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    nixl_status_t rc = backend_->loadRemoteConnInfo(remote_name, conn_info);
    NIXL_INFO << "ProxyRuntime::loadRemoteConnInfo: result=" << rc;
    return rc;
}

nixl_status_t
ProxyRuntime::registerProxyMemView(nixlMemViewH backend_memview,
                                   nixlMemViewH *proxy_memview) {
    return memview_registry_.registerProxyMemView(backend_memview, proxy_memview);
}

nixl_status_t
ProxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    return memview_registry_.unregisterProxyMemView(proxy_memview);
}

nixl_status_t
ProxyRuntime::storeMetadata(nixlMemViewH proxy_memview,
                            const nixl_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

nixl_status_t
ProxyRuntime::storeMetadata(nixlMemViewH proxy_memview,
                            const nixl_remote_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

bool
ProxyRuntime::resolveProxyMemView(nixlMemViewH proxy_memview,
                                  nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemView(proxy_memview, backend_memview);
}

bool
ProxyRuntime::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                    nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemViewId(proxy_memview_id, backend_memview);
}

nixl_status_t
ProxyRuntime::startWorkers() {
    NIXL_INFO << "ProxyRuntime::startWorkers: launching "
              << workers_.size() << " worker thread(s)";
    if (shutdown_word_host_ == nullptr) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }
    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(ProxyControlState::Shutdown),
                     __ATOMIC_RELEASE);
    joinWorkerThreads();
    for (auto &channel : channels_) {
        channel.inflight_requests.clear();
        channel.error_latched = false;
    }

    __atomic_store_n(shutdown_word_host_,
                     static_cast<uint32_t>(ProxyControlState::Running),
                     __ATOMIC_RELEASE);

    uint32_t idx = 0;
    for (auto &worker : workers_) {
        worker->start(idx);
        ++idx;
    }

    NIXL_INFO << "ProxyRuntime::startWorkers: all threads launched";
    return NIXL_SUCCESS;
}

void
ProxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker : workers_) {
        worker->join();
    }
}

nixl_status_t
ProxyRuntime::shutdown() {
    NIXL_INFO << "ProxyRuntime::shutdown: signalling workers to stop";
    if (shutdown_word_host_ != nullptr) {
        __atomic_store_n(shutdown_word_host_,
                         static_cast<uint32_t>(ProxyControlState::Shutdown),
                         __ATOMIC_RELEASE);
    }

    joinWorkerThreads();
    NIXL_INFO << "ProxyRuntime::shutdown: all worker threads joined";

    nixl_status_t backend_status = NIXL_SUCCESS;
    if (backend_ != nullptr) {
        NIXL_INFO << "ProxyRuntime::shutdown: shutting down backend";
        backend_status = backend_->shutdown();
        NIXL_INFO << "ProxyRuntime::shutdown: backend shutdown status=" << backend_status;
    }

    workers_.clear();
    memview_registry_.clear();

    if (device_context_) {
        cudaFreeHost(device_context_);
        device_context_ = nullptr;
    }
    if (shutdown_word_host_) {
        cudaFreeHost(shutdown_word_host_);
        shutdown_word_host_ = nullptr;
        shutdown_word_dev_  = nullptr;
    }
    if (device_channel_views_) {
        cudaFreeHost(device_channel_views_);
        device_channel_views_ = nullptr;
    }

    channels_.clear();
    backend_ = nullptr;
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    return backend_status;
}
