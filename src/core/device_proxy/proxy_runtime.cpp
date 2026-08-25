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
#include <utility>

#include "device/device_allocator.h"

nixlProxyMemViewRegistry::~nixlProxyMemViewRegistry() {
    clear();
}

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               nixlMemViewH *proxy_memview) {
    return registerProxyMemView(backend_memview, {}, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                               const std::vector<void *> &direct_ptrs,
                                               nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    RegistryEntry entry;
    entry.proxy_memview_id = static_cast<uint32_t>(next_proxy_memview_id_);
    entry.backend_memview = backend_memview;

    const size_t direct_ptr_bytes = direct_ptrs.size() * sizeof(void *);
    const size_t allocation_size = sizeof(nixlProxyDeviceMemView) + direct_ptr_bytes;

    nixlDeviceAllocator &allocator = nixlGetDeviceAllocator();
    nixlDeviceMem device_memview_mem;
    if (allocator.allocDeviceMem(allocation_size, device_memview_mem) != NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::register: failed to allocate device memview";
        return NIXL_ERR_BACKEND;
    }
    auto *device_memview = device_memview_mem.as<nixlProxyDeviceMemView>();

    const nixlProxyDeviceMemView host_memview{
        entry.proxy_memview_id, static_cast<uint32_t>(direct_ptrs.size()), device_context_};
    nixl_status_t copy_status =
        allocator.copyHostToDevice(device_memview, &host_memview, sizeof(host_memview));
    if (copy_status == NIXL_SUCCESS && !direct_ptrs.empty()) {
        copy_status = allocator.copyHostToDevice(
            device_memview->direct_ptrs, direct_ptrs.data(), direct_ptr_bytes);
    }
    if (copy_status != NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::register: failed to initialize device memview";
        return NIXL_ERR_BACKEND;
    }

    entry.proxy_memview = device_memview;
    entry.proxy_memview_mem = std::move(device_memview_mem);
    const nixlMemViewH handle = entry.proxy_memview;
    const uint32_t handle_id = entry.proxy_memview_id;
    entries_.push_back(std::move(entry));
    handle_to_id_.emplace(handle, handle_id);

    *proxy_memview = handle;
    ++next_proxy_memview_id_;
    NIXL_DEBUG << "nixlProxyMemViewRegistry::register: backend_mvh=" << backend_memview
               << " -> proxy_id=" << (next_proxy_memview_id_ - 1);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    return prepMemView(nullptr, dlist, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    return prepMemView(dlist, {}, proxy_memview);
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                                      const std::vector<void *> &direct_ptrs,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(nullptr, direct_ptrs, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(nixlMemViewH backend_memview,
                                      const nixl_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(backend_memview, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepMemView(nixlMemViewH backend_memview,
                                      const nixl_remote_meta_dlist_t &dlist,
                                      nixlMemViewH *proxy_memview) {
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlMemViewH registered_proxy_memview = nullptr;
    nixl_status_t status = registerProxyMemView(backend_memview, &registered_proxy_memview);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = storeMetadata(registered_proxy_memview, dlist);
    if (status != NIXL_SUCCESS) {
        unregisterProxyMemView(registered_proxy_memview);
        return status;
    }

    *proxy_memview = registered_proxy_memview;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    entry->state = ProxyMemViewRegEntryState::ENTRY_RETIRED;
    handle_to_id_.erase(proxy_memview);
    releaseDeviceMemView(*entry);
    NIXL_DEBUG << "nixlProxyMemViewRegistry::unregister: proxy_id=" << entry->proxy_memview_id;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemView(nixlMemViewH proxy_memview,
                                              nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

bool
nixlProxyMemViewRegistry::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                                nixlMemViewH &backend_memview) const {
    const RegistryEntry *entry = getEntryForId(proxy_memview_id);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return false;
    }
    backend_memview = entry->backend_memview;
    return true;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }

    fillLocalMetadata(dlist, entry->local_metadata);
    entry->remote_metadata = RemoteMetadata{};
    entry->metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_LOCAL;
    entry->state = ProxyMemViewRegEntryState::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(local): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::storeMetadata(nixlMemViewH proxy_memview,
                                        const nixl_remote_meta_dlist_t &dlist) {
    RegistryEntry *entry = getEntryForHandle(proxy_memview);
    if (entry == nullptr || entry->state == ProxyMemViewRegEntryState::ENTRY_RETIRED) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (dlist.getType() != VRAM_SEG) {
        NIXL_ERROR << "nixlProxyMemViewRegistry::storeMetadata(remote): unsupported mem type "
                   << dlist.getType();
        return NIXL_ERR_INVALID_PARAM;
    }

    fillRemoteMetadata(dlist, entry->remote_metadata);
    entry->local_metadata = LocalMetadata{};
    entry->metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_REMOTE;
    entry->state = ProxyMemViewRegEntryState::ENTRY_READY;

    NIXL_DEBUG << "nixlProxyMemViewRegistry::storeMetadata(remote): proxy_id="
               << entry->proxy_memview_id << " entries=" << dlist.descCount();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::prepareSubmission(const nixlProxySubmission &submission,
                                            nixlBackendProxySubmission &prepared_submission) const {
    bool needs_source = false;
    size_t transfer_size = 0;
    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        needs_source = true;
        transfer_size = submission.size;
        break;
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        transfer_size = sizeof(uint64_t);
        break;
    default:
        NIXL_ERROR << "nixlProxyMemViewRegistry::prepareSubmission: unsupported opcode: "
                   << static_cast<uint32_t>(submission.opcode);
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const ProxyMemViewRegStoredEntry *dst_metadata = nullptr;
    nixl_status_t status = getRemoteEntryForSubmission(submission.dst_proxy_memview_id,
                                                       submission.dst_index,
                                                       submission.dst_offset,
                                                       transfer_size,
                                                       dst_metadata);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    nixlBackendProxySubmission prepared{};
    prepared.op_idx = submission.op_idx;
    prepared.opcode = submission.opcode;
    prepared.channel_id = submission.channel_id;
    prepared.flags = submission.flags;
    prepared.size = transfer_size;
    prepared.value = submission.value;
    prepared.remote_agent = dst_metadata->remote_agent;
    prepared.remote.mem_type = VRAM_SEG;
    prepared.remote.desc = nixlMetaDesc(dst_metadata->base_addr + submission.dst_offset,
                                        transfer_size,
                                        dst_metadata->dev_id,
                                        dst_metadata->metadata);

    if (needs_source) {
        const LocalMetadata *local_metadata = nullptr;
        const ProxyMemViewRegStoredEntry *src_metadata = nullptr;
        status = getLocalEntryForSubmission(submission.src_proxy_memview_id,
                                            submission.src_index,
                                            submission.src_offset,
                                            transfer_size,
                                            local_metadata,
                                            src_metadata);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        prepared.local.mem_type = local_metadata->mem_type;
        prepared.local.desc = nixlMetaDesc(src_metadata->base_addr + submission.src_offset,
                                           transfer_size,
                                           src_metadata->dev_id,
                                           src_metadata->metadata);
    }

    prepared_submission = prepared;
    return NIXL_SUCCESS;
}

void
nixlProxyMemViewRegistry::clear() noexcept {
    for (auto &entry : entries_) {
        entry.state = ProxyMemViewRegEntryState::ENTRY_RETIRED;
        releaseDeviceMemView(entry);
    }
    handle_to_id_.clear();
}

void
nixlProxyMemViewRegistry::releaseDeviceMemView(RegistryEntry &entry) noexcept {
    entry.proxy_memview_mem.reset();
    entry.proxy_memview = nullptr;
}

nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) {
    const auto it = handle_to_id_.find(proxy_memview);
    return it == handle_to_id_.end() ? nullptr : getEntryForId(it->second);
}

const nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForHandle(nixlMemViewH proxy_memview) const {
    const auto it = handle_to_id_.find(proxy_memview);
    return it == handle_to_id_.end() ? nullptr : getEntryForId(it->second);
}

nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) {
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_ ||
        proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

const nixlProxyMemViewRegistry::RegistryEntry *
nixlProxyMemViewRegistry::getEntryForId(uint64_t proxy_memview_id) const {
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_ ||
        proxy_memview_id > entries_.size()) {
        return nullptr;
    }
    return &entries_[proxy_memview_id - 1];
}

nixl_status_t
nixlProxyMemViewRegistry::getRemoteEntryForSubmission(
    uint64_t proxy_memview_id,
    size_t index,
    size_t offset,
    size_t size,
    const ProxyMemViewRegStoredEntry *&entry) const {
    entry = nullptr;

    const RegistryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != ProxyMemViewRegEntryState::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst not ready"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadata_kind != ProxyMemViewRegMetadataKind::METADATA_KIND_REMOTE) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst metadata kind invalid"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &remote_metadata = registry_entry->remote_metadata;
    if (index >= remote_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const ProxyMemViewRegStoredEntry &remote_entry = remote_metadata.entries[index];
    if (!rangeFits(remote_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_entry.remote_agent.empty() || remote_entry.remote_agent == nixl_null_agent) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: dst remote agent invalid"
                   << " dst_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    entry = &remote_entry;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyMemViewRegistry::getLocalEntryForSubmission(
    uint64_t proxy_memview_id,
    size_t index,
    size_t offset,
    size_t size,
    const LocalMetadata *&metadata,
    const ProxyMemViewRegStoredEntry *&entry) const {
    metadata = nullptr;
    entry = nullptr;

    const RegistryEntry *registry_entry = getEntryForId(proxy_memview_id);
    if (registry_entry == nullptr ||
        registry_entry->state != ProxyMemViewRegEntryState::ENTRY_READY) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src not ready"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_NOT_FOUND;
    }
    if (registry_entry->metadata_kind != ProxyMemViewRegMetadataKind::METADATA_KIND_LOCAL) {
        NIXL_DEBUG << "nixlProxyMemViewRegistry::prepareSubmission: src metadata kind invalid"
                   << " src_proxy_id=" << proxy_memview_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &local_metadata = registry_entry->local_metadata;
    if (index >= local_metadata.entries.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const ProxyMemViewRegStoredEntry &local_entry = local_metadata.entries[index];
    if (!rangeFits(local_entry, offset, size)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    metadata = &local_metadata;
    entry = &local_entry;
    return NIXL_SUCCESS;
}

bool
nixlProxyMemViewRegistry::rangeFits(const ProxyMemViewRegStoredEntry &entry,
                                    size_t offset,
                                    size_t size) {
    return offset <= entry.len && size <= entry.len - offset;
}

void
nixlProxyMemViewRegistry::fillLocalMetadata(const nixl_meta_dlist_t &dlist, LocalMetadata &out) {
    out = LocalMetadata{};
    out.mem_type = dlist.getType();
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(
            ProxyMemViewRegStoredEntry{desc.addr, desc.len, desc.devId, desc.metadataP});
    }
}

void
nixlProxyMemViewRegistry::fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist,
                                             RemoteMetadata &out) {
    out = RemoteMetadata{};
    out.entries.reserve(dlist.descCount());
    for (const auto &desc : dlist) {
        out.entries.push_back(ProxyMemViewRegStoredEntry{
            desc.addr, desc.len, desc.devId, desc.metadataP, desc.remoteAgent});
    }
}

nixl_status_t
nixlProxyChannelState::allocate(uint32_t depth,
                                nixlProxyControlBuffer *control_slots,
                                size_t control_slot_index) {
    NIXL_INFO << "nixlProxyChannelState::allocate: depth=" << depth
              << " control_slot_index=" << control_slot_index;
    if (depth == 0 || control_slots == nullptr ||
        control_slots->devicePtr(control_slot_index) == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    ring_depth_ = depth;
    control_slots_ = control_slots;
    control_slot_index_ = control_slot_index;
    consumer_idx_dev_ = control_slots_->devicePtr(control_slot_index_);
    consumer_idx_shadow_ = 0;

    nixlDeviceAllocator &allocator = nixlGetDeviceAllocator();
    if (allocator.allocDeviceMem(sizeof(nixlProxyWorkRing), work_ring_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), producer_idx_mem_) != NIXL_SUCCESS ||
        allocator.allocDeviceMem(sizeof(uint64_t), consumer_idx_cache_mem_) != NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxySubmission) * depth, records_mem_) !=
            NIXL_SUCCESS ||
        allocator.allocMappedHostMem(sizeof(nixlProxyCompletionSlot), completion_slot_mem_) !=
            NIXL_SUCCESS) {
        NIXL_ERROR << "nixlProxyChannelState::allocate: device allocation failed";
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    nixlProxySubmission *records_host = recordsHost();
    for (uint32_t i = 0; i < depth; ++i) {
        records_host[i] = nixlProxySubmission{};
    }
    if (allocator.memsetDeviceMem(producer_idx_mem_.get(), 0, sizeof(uint64_t)) != NIXL_SUCCESS ||
        allocator.memsetDeviceMem(consumer_idx_cache_mem_.get(), 0, sizeof(uint64_t)) !=
            NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    if (publishConsumerIdx(0) != NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    submit_idx_ = 0;
    completionSlotHost()->next_status = NIXL_IN_PROG;
    __atomic_store_n(&completionSlotHost()->completed_idx, uint64_t{0}, __ATOMIC_RELEASE);
    nixlProxyWorkRing work_ring{
        records_mem_.asDev<nixlProxySubmission>(),
        producer_idx_mem_.as<uint64_t>(),
        consumer_idx_dev_,
        consumer_idx_cache_mem_.as<uint64_t>(),
        depth,
    };
    if (allocator.copyHostToDevice(work_ring_mem_.get(), &work_ring, sizeof(work_ring)) !=
        NIXL_SUCCESS) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    device_view = nixlProxyChannelView{work_ring_mem_.as<nixlProxyWorkRing>(),
                                       completion_slot_mem_.asDev<nixlProxyCompletionSlot>()};

    inflight_slots_.assign(depth, nixlProxyRequestState{});
    NIXL_INFO << "nixlProxyChannelState::allocate: ready"
              << " work_ring(dev)=" << work_ring_mem_.get() << " records=" << recordsHost()
              << " records(dev)=" << records_mem_.devPtr()
              << " producer_idx(dev)=" << producer_idx_mem_.get()
              << " consumer_idx(shadow)=" << consumer_idx_shadow_
              << " consumer_idx(dev)=" << consumer_idx_dev_
              << " consumer_idx_cache(dev)=" << consumer_idx_cache_mem_.get()
              << " completion_slot(host)=" << completionSlotHost()
              << " completion_slot(dev)=" << completion_slot_mem_.devPtr();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyChannelState::publishConsumerIdx(uint64_t value) noexcept {
    if (control_slots_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const nixl_status_t status = control_slots_->writeSlot(control_slot_index_, value);
    if (status == NIXL_SUCCESS) {
        consumer_idx_shadow_ = value;
    }
    return status;
}

void
nixlProxyChannelState::deallocate() noexcept {
    completion_slot_mem_.reset();
    records_mem_.reset();
    producer_idx_mem_.reset();
    consumer_idx_cache_mem_.reset();
    work_ring_mem_.reset();
    consumer_idx_dev_ = nullptr;
    control_slots_ = nullptr;
    control_slot_index_ = 0;
    consumer_idx_shadow_ = 0;
    inflight_slots_.clear();
    submit_idx_ = 0;
    ring_depth_ = 0;
    device_view = nixlProxyChannelView{};
}

nixlProxyRuntime::nixlProxyRuntime() = default;

nixlProxyRuntime::~nixlProxyRuntime() {
    if (backend_) {
        shutdown();
    }
}

nixl_status_t
nixlProxyRuntime::init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
                       uint32_t max_peers,
                       uint32_t channel_count,
                       uint32_t worker_count,
                       uint64_t pthr_delay_us) {
    NIXL_INFO << "ProxyRuntime::init: max_peers=" << max_peers << " channel_count=" << channel_count
              << " worker_count=" << worker_count << " pthr_delay_us=" << pthr_delay_us
              << " backend=" << backend.get();
    if (backend == nullptr || max_peers == 0 || channel_count == 0 || worker_count == 0) {
        NIXL_ERROR << "ProxyRuntime::init: invalid params";
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = std::move(backend);
    memview_registry_.clear();

    const uint32_t effective_worker_count = std::min(worker_count, channel_count);
    NIXL_INFO << "ProxyRuntime::init: effective worker_count=" << effective_worker_count
              << " (clamped to channel_count)";

    nixl_status_t rc = backend_->init(effective_worker_count, channel_count, max_peers);
    if ((rc != NIXL_SUCCESS) && (rc != NIXL_ERR_NOT_SUPPORTED)) {
        NIXL_ERROR << "ProxyRuntime::init: backend init failed: " << rc;
        backend_.reset();
        return rc;
    }
    if (rc == NIXL_ERR_NOT_SUPPORTED) {
        NIXL_INFO << "ProxyRuntime::init: backend init hook not supported; continuing";
    }

    const size_t channel_slots = static_cast<size_t>(max_peers) * channel_count;
    rc = control_slots_.allocate(kProxyCiSlotBase + channel_slots);
    if (rc != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::init: failed to create GPU-visible control slab";
        shutdown();
        return rc;
    }
    shutdown_word_dev_ = control_slots_.devicePtr(kProxyShutdownSlot);
    channels_.resize(channel_slots);
    device_channel_views_.resize(channel_slots);
    for (uint32_t channel_idx = 0; channel_idx < channel_count; channel_idx++) {
        for (uint32_t peer_idx = 0; peer_idx < max_peers; peer_idx++) {
            const size_t slot = static_cast<size_t>(channel_idx) * max_peers + peer_idx;
            rc = channels_[slot].allocate(ring_depth_, &control_slots_, kProxyCiSlotBase + slot);
            if (rc != NIXL_SUCCESS) {
                shutdown();
                return rc;
            }
            device_channel_views_[slot] = channels_[slot].device_view;
        }
    }

    nixlDeviceAllocator &allocator = nixlGetDeviceAllocator();
    if (allocator.allocDeviceMem(sizeof(nixlProxyChannelView) * channel_slots,
                                 device_channel_views_mem_) != NIXL_SUCCESS ||
        allocator.copyHostToDevice(device_channel_views_mem_.get(),
                                   device_channel_views_.data(),
                                   sizeof(nixlProxyChannelView) * channel_slots) != NIXL_SUCCESS) {
        shutdown();
        return NIXL_ERR_BACKEND;
    }

    nixlProxyDeviceContextData device_context{device_channel_views_mem_.as<nixlProxyChannelView>(),
                                              max_peers,
                                              channel_count,
                                              shutdown_word_dev_};
    if (allocator.allocDeviceMem(sizeof(nixlProxyDeviceContextData), device_context_mem_) !=
            NIXL_SUCCESS ||
        allocator.copyHostToDevice(
            device_context_mem_.get(), &device_context, sizeof(device_context)) != NIXL_SUCCESS) {
        shutdown();
        return NIXL_ERR_BACKEND;
    }
    memview_registry_.setDeviceContext(deviceContext());

    workers_.clear();
    workers_.reserve(effective_worker_count);
    workers_started_ = false;

    for (uint32_t worker_idx = 0; worker_idx < effective_worker_count; worker_idx++) {
        NIXL_INFO << "ProxyRuntime::init: worker " << worker_idx
                  << " owns channel(s) where channel_id % " << effective_worker_count
                  << " == " << worker_idx << "; handles all dest rings of those channels";
        workers_.push_back(std::make_unique<ProxyWorker>(backend_.get(),
                                                         &memview_registry_,
                                                         &shutdown_state_,
                                                         channels_.data(),
                                                         max_peers,
                                                         channel_count,
                                                         worker_idx,
                                                         effective_worker_count,
                                                         pthr_delay_us));
    }

    NIXL_INFO << "ProxyRuntime::init: complete — " << max_peers << " peers, " << channel_count
              << " channels (rings per dest), " << effective_worker_count
              << " workers, device_context(dev)=" << deviceContext();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlProxyRuntime::loadRemoteConnInfo(const std::string &remote_name, const nixl_blob_t &conn_info) {
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
nixlProxyRuntime::registerProxyMemView(nixlMemViewH backend_memview, nixlMemViewH *proxy_memview) {
    return memview_registry_.registerProxyMemView(backend_memview, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(const nixl_remote_meta_dlist_t &dlist, nixlMemViewH *proxy_memview) {
    std::vector<void *> direct_ptrs;
    if (backend_ != nullptr) {
        const nixl_status_t resolve_status = backend_->resolveDirectPointers(dlist, direct_ptrs);
        if (resolve_status == NIXL_ERR_NOT_SUPPORTED) {
            direct_ptrs.clear();
        } else if (resolve_status != NIXL_SUCCESS) {
            return resolve_status;
        }
    }

    return memview_registry_.prepMemView(dlist, direct_ptrs, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(nixlMemViewH backend_memview,
                              const nixl_meta_dlist_t &dlist,
                              nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(backend_memview, dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::prepMemView(nixlMemViewH backend_memview,
                              const nixl_remote_meta_dlist_t &dlist,
                              nixlMemViewH *proxy_memview) {
    return memview_registry_.prepMemView(backend_memview, dlist, proxy_memview);
}

nixl_status_t
nixlProxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    return memview_registry_.unregisterProxyMemView(proxy_memview);
}

nixl_status_t
nixlProxyRuntime::storeMetadata(nixlMemViewH proxy_memview, const nixl_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

nixl_status_t
nixlProxyRuntime::storeMetadata(nixlMemViewH proxy_memview, const nixl_remote_meta_dlist_t &dlist) {
    return memview_registry_.storeMetadata(proxy_memview, dlist);
}

bool
nixlProxyRuntime::resolveProxyMemView(nixlMemViewH proxy_memview,
                                      nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemView(proxy_memview, backend_memview);
}

bool
nixlProxyRuntime::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                        nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemViewId(proxy_memview_id, backend_memview);
}

nixl_status_t
nixlProxyRuntime::startWorkers() {
    NIXL_INFO << "ProxyRuntime::startWorkers: launching " << workers_.size() << " worker thread(s)";
    if (!control_slots_.allocated()) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: runtime not initialized";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (workers_started_) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: workers already started";
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_status_t publish_status = control_slots_.writeSlot(
        kProxyShutdownSlot, static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING));
    if (publish_status != NIXL_SUCCESS) {
        NIXL_ERROR << "ProxyRuntime::startWorkers: failed to publish RUNNING state";
        return publish_status;
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING),
                          std::memory_order_release);

    for (auto &worker : workers_) {
        worker->start();
    }
    workers_started_ = true;

    NIXL_INFO << "ProxyRuntime::startWorkers: all threads launched";
    return NIXL_SUCCESS;
}

void
nixlProxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker : workers_) {
        worker->join();
    }
}

nixl_status_t
nixlProxyRuntime::shutdown() {
    NIXL_INFO << "ProxyRuntime::shutdown: signalling workers to stop";
    nixl_status_t shutdown_signal_status = NIXL_SUCCESS;
    if (control_slots_.allocated()) {
        shutdown_signal_status = control_slots_.writeSlot(
            kProxyShutdownSlot, static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN));
        if (shutdown_signal_status != NIXL_SUCCESS) {
            NIXL_ERROR << "ProxyRuntime::shutdown: failed to publish SHUTDOWN state";
        }
    }
    shutdown_state_.store(static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN),
                          std::memory_order_release);

    joinWorkerThreads();
    workers_started_ = false;
    NIXL_INFO << "ProxyRuntime::shutdown: all worker threads joined";

    if (backend_ != nullptr) {
        size_t released = 0;
        for (auto &channel : channels_) {
            if (channel.ring_depth_ == 0 || channel.consumer_idx_dev_ == nullptr) {
                continue;
            }

            const uint64_t consumer_idx = channel.consumer_idx_shadow_;
            for (uint64_t idx = consumer_idx; idx < channel.submit_idx_; ++idx) {
                nixlProxyRequestState &inflight =
                    channel.inflight_slots_[idx % channel.ring_depth_];
                if (inflight.status == NIXL_IN_PROG && inflight.backend_request) {
                    backend_->releaseRequest(inflight.backend_request);
                    ++released;
                }
                inflight = nixlProxyRequestState{};
            }
            channel.submit_idx_ = consumer_idx;
        }
        if (released != 0) {
            NIXL_INFO << "ProxyRuntime::shutdown: released " << released
                      << " pending backend request(s)";
        }
    }

    nixl_status_t backend_status = NIXL_SUCCESS;
    if (backend_ != nullptr) {
        NIXL_INFO << "ProxyRuntime::shutdown: shutting down backend";
        backend_status = backend_->shutdown();
        NIXL_INFO << "ProxyRuntime::shutdown: backend shutdown status=" << backend_status;
        if (backend_status == NIXL_ERR_NOT_SUPPORTED) {
            backend_status = NIXL_SUCCESS;
        }
    }

    workers_.clear();
    memview_registry_.clear();
    memview_registry_.setDeviceContext(nullptr);

    device_context_mem_.reset();
    shutdown_word_dev_ = nullptr;
    device_channel_views_mem_.reset();
    device_channel_views_.clear();

    channels_.clear();
    control_slots_.deallocate();
    backend_.reset();
    NIXL_INFO << "ProxyRuntime::shutdown: complete";
    if (backend_status != NIXL_SUCCESS) {
        return backend_status;
    }
    if (shutdown_signal_status != NIXL_SUCCESS) {
        return shutdown_signal_status;
    }
    return NIXL_SUCCESS;
}
