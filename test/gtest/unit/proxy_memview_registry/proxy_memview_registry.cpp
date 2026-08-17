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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_memview_registry {

    class ProxyMemViewRegistryTest : public testing::Test {
    protected:
        class DummyBackendMD : public nixlBackendMD {
        public:
            DummyBackendMD() : nixlBackendMD(false) {}
        };

        nixlProxyMemViewRegistry registry_;
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;

        struct MemViewPair {
            nixlMemViewH src = nullptr;
            nixlMemViewH dst = nullptr;
        };

        nixlMemViewH
        makeFakeBackendHandle(uint64_t id) {
            return reinterpret_cast<nixlMemViewH>(id);
        }

        static uint32_t
        proxyMemViewId(nixlMemViewH proxy_memview) {
            if (proxy_memview == nullptr) {
                return 0;
            }
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview.proxy_memview_id;
        }

        static nixlProxyDeviceMemView
        copyDeviceMemView(nixlMemViewH proxy_memview) {
            nixlProxyDeviceMemView device_memview{};
            EXPECT_EQ(
                cudaMemcpy(
                    &device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
                cudaSuccess);
            return device_memview;
        }

        static std::vector<void *>
        copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
            std::vector<void *> direct_ptrs(count, nullptr);
            if (count != 0) {
                auto *direct_ptrs_dev =
                    static_cast<nixlProxyDeviceMemView *>(proxy_memview)->direct_ptrs;
                EXPECT_EQ(cudaMemcpy(direct_ptrs.data(),
                                     direct_ptrs_dev,
                                     sizeof(void *) * count,
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
            }
            return direct_ptrs;
        }

        nixl_meta_dlist_t
        makeLocalMetadata(uintptr_t base_addr, uint64_t dev_id = 0, uint64_t len = 64) {
            nixl_meta_dlist_t dlist(DRAM_SEG);
            dlist.addDesc(nixlMetaDesc(base_addr, len, dev_id, &local_md_));
            return dlist;
        }

        nixl_remote_meta_dlist_t
        makeRemoteMetadata(uintptr_t base_addr,
                           const std::string &remote_agent = "peer",
                           uint64_t dev_id = 0,
                           nixl_mem_t mem_type = VRAM_SEG,
                           uint64_t len = 64) {
            nixl_remote_meta_dlist_t dlist(mem_type);
            nixlRemoteMetaDesc desc(remote_agent);
            desc.addr = base_addr;
            desc.len = len;
            desc.devId = dev_id;
            desc.metadataP = &remote_md_;
            dlist.addDesc(desc);
            return dlist;
        }

        void
        registerPair(MemViewPair *pair) {
            ASSERT_NE(pair, nullptr);
            ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &pair->src),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &pair->dst),
                      NIXL_SUCCESS);
        }

        void
        registerRemote(nixlMemViewH *dst_proxy) {
            ASSERT_NE(dst_proxy, nullptr);
            ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), dst_proxy),
                      NIXL_SUCCESS);
        }

        void
        storePair(const MemViewPair &pair,
                  uintptr_t src_addr = 0x1000,
                  uintptr_t dst_addr = 0x2000,
                  const std::string &remote_agent = "peer",
                  uint64_t src_dev_id = 0,
                  uint64_t dst_dev_id = 0,
                  uint64_t src_len = 64,
                  uint64_t dst_len = 64) {
            ASSERT_EQ(registry_.storeMetadata(pair.src,
                                              makeLocalMetadata(src_addr, src_dev_id, src_len)),
                      NIXL_SUCCESS);
            ASSERT_EQ(registry_.storeMetadata(
                          pair.dst,
                          makeRemoteMetadata(dst_addr, remote_agent, dst_dev_id, VRAM_SEG, dst_len)),
                      NIXL_SUCCESS);
        }

        MemViewPair
        makeStoredPair(uintptr_t src_addr = 0x1000,
                       uintptr_t dst_addr = 0x2000,
                       const std::string &remote_agent = "peer",
                       uint64_t src_dev_id = 0,
                       uint64_t dst_dev_id = 0,
                       uint64_t src_len = 64,
                       uint64_t dst_len = 64) {
            MemViewPair pair;
            registerPair(&pair);
            storePair(
                pair, src_addr, dst_addr, remote_agent, src_dev_id, dst_dev_id, src_len, dst_len);
            return pair;
        }

        static nixlProxySubmission
        makePutSubmission(const MemViewPair &pair,
                          uint64_t src_offset = 0,
                          uint64_t dst_offset = 0,
                          uint64_t size = 16) {
            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.src_proxy_memview_id = proxyMemViewId(pair.src);
            submission.src_offset = src_offset;
            submission.dst_proxy_memview_id = proxyMemViewId(pair.dst);
            submission.dst_offset = dst_offset;
            submission.size = size;
            return submission;
        }

        static nixlProxySubmission
        makeAtomicAddSubmission(nixlMemViewH dst_proxy, uint64_t dst_offset = 0) {
            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
            submission.dst_offset = dst_offset;
            return submission;
        }
    };

    struct InvalidRemoteAgentConfig {
        const char *name;
        std::string agent;
    };

    class ProxyMemViewRegistryInvalidRemoteAgentTest
        : public ProxyMemViewRegistryTest,
          public testing::WithParamInterface<InvalidRemoteAgentConfig> {};

    struct PutRangeConfig {
        const char *name;
        uint64_t src_offset;
        uint64_t dst_offset;
        uint64_t size;
        nixl_status_t expected_status;
        bool preserves_output;
    };

    class ProxyMemViewRegistryPutRangeTest
        : public ProxyMemViewRegistryTest,
          public testing::WithParamInterface<PutRangeConfig> {};

    struct LargePutConfig {
        const char *name;
        uint64_t metadata_len;
        uint64_t src_offset;
        uint64_t dst_offset;
        uint64_t size;
        bool check_offsets;
    };

    class ProxyMemViewRegistryLargePutTest
        : public ProxyMemViewRegistryTest,
          public testing::WithParamInterface<LargePutConfig> {};

    TEST_F(ProxyMemViewRegistryTest, RegisterSingle) {
        nixlMemViewH proxy_handle = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), &proxy_handle),
                  NIXL_SUCCESS);
        EXPECT_NE(proxy_handle, nullptr);

        const nixlProxyDeviceMemView device_memview = copyDeviceMemView(proxy_handle);
        EXPECT_EQ(device_memview.proxy_memview_id, 1u);
        EXPECT_EQ(device_memview.direct_ptr_count, 0u);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterNullOutputReturnsError) {
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(100), nullptr),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RegisterMultipleAssignsUniqueIds) {
        nixlMemViewH h1 = nullptr, h2 = nullptr, h3 = nullptr;
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &h1), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &h2), NIXL_SUCCESS);
        EXPECT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(30), &h3), NIXL_SUCCESS);

        EXPECT_NE(h1, h2);
        EXPECT_NE(h2, h3);
        EXPECT_NE(h1, h3);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveByHandleBeforeMetadataPublish) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(proxy_handle, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveById) {
        auto backend = makeFakeBackendHandle(42);
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(backend, &proxy_handle), NIXL_SUCCESS);

        auto proxy_id = proxyMemViewId(proxy_handle);
        nixlMemViewH resolved = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemViewId(proxy_id, resolved));
        EXPECT_EQ(resolved, backend);
    }

    TEST_F(ProxyMemViewRegistryTest, ResolveMultiple) {
        auto b1 = makeFakeBackendHandle(10), b2 = makeFakeBackendHandle(20);
        nixlMemViewH h1 = nullptr, h2 = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(b1, &h1), NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(b2, &h2), NIXL_SUCCESS);

        nixlMemViewH r1 = nullptr, r2 = nullptr;
        EXPECT_TRUE(registry_.resolveProxyMemView(h1, r1));
        EXPECT_TRUE(registry_.resolveProxyMemView(h2, r2));
        EXPECT_EQ(r1, b1);
        EXPECT_EQ(r2, b2);
    }

    TEST_F(ProxyMemViewRegistryTest, SubmissionRecordStaysPackedTo64Bytes) {
        EXPECT_EQ(sizeof(nixlProxySubmission), 64u);
        EXPECT_EQ(alignof(nixlProxySubmission), 64u);
        EXPECT_EQ(offsetof(nixlProxySubmission, op_idx), 0u);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRequiresReadyEntries) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_ERR_NOT_FOUND);
    }

    TEST_F(ProxyMemViewRegistryTest, ReadyEntriesProducePreparedTransportDescriptors) {
        const MemViewPair pair = makeStoredPair(0x1000, 0x2000, "remote-agent");
        auto submission = makePutSubmission(pair, 5, 9);
        submission.op_idx = 7;
        submission.channel_id = 3;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.op_idx, 7u);
        EXPECT_EQ(prepared_submission.channel_id, 3u);
        EXPECT_EQ(prepared_submission.local.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared_submission.local.desc.addr, 0x1005u);
        EXPECT_EQ(prepared_submission.local.desc.len, 16u);
        EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared_submission.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared_submission.remote_agent, "remote-agent");
    }

    TEST_P(ProxyMemViewRegistryLargePutTest, PrepareSubmissionAcceptsLargeValues) {
        const auto &config = GetParam();
        const MemViewPair pair =
            makeStoredPair(0x1000, 0x2000, "peer", 0, 0, config.metadata_len, config.metadata_len);
        const auto submission =
            makePutSubmission(pair, config.src_offset, config.dst_offset, config.size);

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.size, config.size);
        EXPECT_EQ(prepared_submission.local.desc.len, config.size);
        EXPECT_EQ(prepared_submission.remote.desc.len, config.size);
        if (config.check_offsets) {
            EXPECT_EQ(prepared_submission.local.desc.addr, uintptr_t{0x1000} + config.src_offset);
            EXPECT_EQ(prepared_submission.remote.desc.addr, uintptr_t{0x2000} + config.dst_offset);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        LargeValues,
        ProxyMemViewRegistryLargePutTest,
        testing::Values(LargePutConfig{"PrepareSubmissionAccepts64BitOffsets",
                                       (uint64_t{1} << 32) + 80,
                                       (uint64_t{1} << 32) + 16,
                                       (uint64_t{1} << 32) + 16,
                                       32,
                                       true},
                        LargePutConfig{"PrepareSubmissionAccepts64BitSize",
                                       (uint64_t{1} << 32) + 64,
                                       0,
                                       0,
                                       (uint64_t{1} << 32) + 64,
                                       false}),
        [](const testing::TestParamInfo<LargePutConfig> &info) { return info.param.name; });

    TEST_F(ProxyMemViewRegistryTest, StoreRemoteMetadataRejectsNonVram) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);

        EXPECT_EQ(registry_.storeMetadata(dst_proxy,
                                          makeRemoteMetadata(0x2000, "remote-agent", 0, DRAM_SEG)),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepMemViewProducesReadyEntries) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepMemView(makeLocalMetadata(0x1000), &src_proxy), NIXL_SUCCESS);
        ASSERT_EQ(registry_.prepMemView(makeRemoteMetadata(0x2000), &dst_proxy), NIXL_SUCCESS);

        nixlMemViewH resolved = makeFakeBackendHandle(42);
        EXPECT_TRUE(registry_.resolveProxyMemView(src_proxy, resolved));
        EXPECT_EQ(resolved, nullptr);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = 4;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 8;
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
        EXPECT_EQ(prepared_submission.local.desc.len, 16u);
        EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
        EXPECT_EQ(prepared_submission.remote.desc.len, 16u);
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
    }

    TEST_F(ProxyMemViewRegistryTest, PrepRemoteMemViewStoresDirectPointers) {
        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        nixlRemoteMetaDesc first("peer0");
        first.addr = 0x2000;
        first.len = 64;
        first.devId = 0;
        first.metadataP = &remote_md_;
        remote_dlist.addDesc(first);
        nixlRemoteMetaDesc second("peer1");
        second.addr = 0x3000;
        second.len = 64;
        second.devId = 1;
        second.metadataP = &remote_md_;
        remote_dlist.addDesc(second);

        std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xfeed0000}), nullptr};
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.prepMemView(remote_dlist, direct_ptrs, &dst_proxy), NIXL_SUCCESS);

        const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
        EXPECT_EQ(device_memview.proxy_memview_id, proxyMemViewId(dst_proxy));
        EXPECT_EQ(device_memview.direct_ptr_count, direct_ptrs.size());
        EXPECT_EQ(copyDirectPointers(dst_proxy, direct_ptrs.size()), direct_ptrs);
    }

    TEST_P(ProxyMemViewRegistryPutRangeTest, PrepareSubmissionValidatesPutRanges) {
        const MemViewPair pair = makeStoredPair();
        const auto &config = GetParam();
        const auto submission =
            makePutSubmission(pair, config.src_offset, config.dst_offset, config.size);

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), config.expected_status);
        if (config.expected_status == NIXL_SUCCESS) {
            EXPECT_EQ(prepared_submission.local.desc.addr, 0x1000u + config.src_offset);
            EXPECT_EQ(prepared_submission.local.desc.len, config.size);
            EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2000u + config.dst_offset);
            EXPECT_EQ(prepared_submission.remote.desc.len, config.size);
        } else if (config.preserves_output) {
            EXPECT_EQ(prepared_submission.op_idx, 123u);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        RangeCases,
        ProxyMemViewRegistryPutRangeTest,
        testing::Values(PutRangeConfig{"AtDescriptorBoundary", 48, 48, 16, NIXL_SUCCESS, false},
                        PutRangeConfig{"SourceOutsideDescriptor", 60, 0, 8, NIXL_ERR_INVALID_PARAM, true},
                        PutRangeConfig{"DestinationOutsideDescriptor", 0, 60, 8, NIXL_ERR_INVALID_PARAM, false},
                        PutRangeConfig{"Overflow", 0,
                                       std::numeric_limits<uint32_t>::max(), 1,
                                       NIXL_ERR_INVALID_PARAM, false}),
        [](const testing::TestParamInfo<PutRangeConfig> &info) { return info.param.name; });

    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionRejectsUnsupportedOpcode) {
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = static_cast<nixl_proxy_opcode_t>(99);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        prepared_submission.op_idx = 123;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_NOT_SUPPORTED);
        EXPECT_EQ(prepared_submission.op_idx, 123u);
    }

    TEST_F(ProxyMemViewRegistryTest, PreparedDescriptorsPreserveDeviceIds) {
        const MemViewPair pair = makeStoredPair(0x1000, 0x2000, "peer", 7, 11);
        const auto submission = makePutSubmission(pair, 0, 0, 8);

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.local.desc.devId, 7u);
        EXPECT_EQ(prepared_submission.remote.desc.devId, 11u);
    }

    TEST_F(ProxyMemViewRegistryTest, AtomicAddUsesCounterSizeForDestinationBounds) {
        nixlMemViewH dst_proxy = nullptr;
        registerRemote(&dst_proxy);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);
        auto submission = makeAtomicAddSubmission(dst_proxy, 56);

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.size, sizeof(uint64_t));
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2038u);
        EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));

        submission.dst_offset = 60;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, ReadyRemoteEntryProducesAtomicPreparedDescriptor) {
        nixlMemViewH dst_proxy = nullptr;
        registerRemote(&dst_proxy);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, "remote-agent")),
                  NIXL_SUCCESS);
        auto submission = makeAtomicAddSubmission(dst_proxy, 9);
        submission.op_idx = 7;
        submission.channel_id = 3;
        submission.size = sizeof(uint64_t);
        submission.value = 42;

        nixlBackendProxySubmission prepared_submission;
        ASSERT_EQ(registry_.prepareSubmission(submission, prepared_submission), NIXL_SUCCESS);
        EXPECT_EQ(prepared_submission.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(prepared_submission.op_idx, 7u);
        EXPECT_EQ(prepared_submission.channel_id, 3u);
        EXPECT_EQ(prepared_submission.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared_submission.remote.desc.len, sizeof(uint64_t));
        EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared_submission.remote_agent, "remote-agent");
        EXPECT_EQ(prepared_submission.value, 42u);
    }

    TEST_P(ProxyMemViewRegistryInvalidRemoteAgentTest, PrepareSubmissionRejectsInvalidRemoteAgent) {
        nixlMemViewH dst_proxy = nullptr;
        registerRemote(&dst_proxy);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000, GetParam().agent)),
                  NIXL_SUCCESS);
        const auto submission = makeAtomicAddSubmission(dst_proxy);

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    INSTANTIATE_TEST_SUITE_P(
        InvalidAgents,
        ProxyMemViewRegistryInvalidRemoteAgentTest,
        testing::Values(InvalidRemoteAgentConfig{"Empty", ""},
                        InvalidRemoteAgentConfig{"Null", nixl_null_agent}),
        [](const testing::TestParamInfo<InvalidRemoteAgentConfig> &info) {
            return info.param.name;
        });

    TEST_F(ProxyMemViewRegistryTest, MetadataKindMustMatchSubmissionRole) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeRemoteMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeLocalMetadata(0x2000)), NIXL_SUCCESS);

        nixlProxySubmission submission{};
        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.size = 16;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(submission, prepared_submission),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyMemViewRegistryTest, RetiredEntriesStopFutureDispatchButKeepOtherEntriesUsable) {
        nixlMemViewH src_proxy = nullptr;
        nixlMemViewH dst_proxy = nullptr;
        nixlMemViewH other_proxy = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(20), &dst_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(30), &other_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(src_proxy, makeLocalMetadata(0x1000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(dst_proxy, makeRemoteMetadata(0x2000)), NIXL_SUCCESS);
        ASSERT_EQ(registry_.storeMetadata(other_proxy, makeRemoteMetadata(0x3000)), NIXL_SUCCESS);
        const uint32_t src_proxy_id = proxyMemViewId(src_proxy);
        const uint32_t dst_proxy_id = proxyMemViewId(dst_proxy);
        const uint32_t other_proxy_id = proxyMemViewId(other_proxy);

        ASSERT_EQ(registry_.unregisterProxyMemView(dst_proxy), NIXL_SUCCESS);
        EXPECT_EQ(registry_.unregisterProxyMemView(dst_proxy), NIXL_ERR_INVALID_PARAM);

        nixlProxySubmission retired_submission{};
        retired_submission.opcode = nixl_proxy_opcode_t::PUT;
        retired_submission.src_proxy_memview_id = src_proxy_id;
        retired_submission.dst_proxy_memview_id = dst_proxy_id;
        retired_submission.size = 8;

        nixlBackendProxySubmission prepared_submission;
        EXPECT_EQ(registry_.prepareSubmission(retired_submission, prepared_submission),
                  NIXL_ERR_NOT_FOUND);

        nixlProxySubmission live_submission{};
        live_submission.opcode = nixl_proxy_opcode_t::PUT;
        live_submission.src_proxy_memview_id = src_proxy_id;
        live_submission.dst_proxy_memview_id = other_proxy_id;
        live_submission.size = 8;

        EXPECT_EQ(registry_.prepareSubmission(live_submission, prepared_submission), NIXL_SUCCESS);
    }

    TEST_F(ProxyMemViewRegistryTest, StoreMetadataRejectsRetiredEntries) {
        nixlMemViewH proxy_handle = nullptr;
        ASSERT_EQ(registry_.registerProxyMemView(makeFakeBackendHandle(10), &proxy_handle),
                  NIXL_SUCCESS);
        ASSERT_EQ(registry_.unregisterProxyMemView(proxy_handle), NIXL_SUCCESS);
        EXPECT_EQ(registry_.storeMetadata(proxy_handle, makeLocalMetadata(0x1000)),
                  NIXL_ERR_NOT_FOUND);
    }

} // namespace proxy_memview_registry
} // namespace gtest
