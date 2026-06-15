# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if not cuda_dep.found()
    warning('CUDA not found, skipping device pingpong benchmark')
    subdir_done()
endif

if not gpu_device_api_supported
    warning('GPU Device API backend is disabled or unavailable, skipping device pingpong benchmark')
    subdir_done()
endif

if not ucx_dep.found()
    warning('UCX backend dependency not available, skipping device pingpong benchmark')
    subdir_done()
endif

nixl_dep = declare_dependency(
    link_with: nixl_lib,
    include_directories: nixl_inc_dirs,
)
nixl_lib_dir  = join_paths(meson.project_build_root(), 'src', 'core')
nixl_infra_dir = join_paths(meson.project_build_root(), 'src', 'infra')

pingpong_sources = [
    'bench_main.cpp',
    'bench_host.cpp',
    'bench_kernel.cu',
]

pingpong_inc_dirs = [
    nixl_inc_dirs,
    nixl_gpu_device_api_inc_dirs,
    include_directories('.'),
]

if gpu_device_api_is_proxy
    pingpong_inc_dirs += [nixl_gpu_proxy_inc_dirs]
endif

pingpong_cpp_args = [
    '-DHAVE_CUDA',
    '-Wno-deprecated-declarations',
]

pingpong_cuda_args = [
    '-DHAVE_CUDA',
    '-gencode', 'arch=compute_89,code=sm_89',
    '--expt-relaxed-constexpr',
    '-Xcompiler', '-Wno-deprecated-declarations',
    '-Xcompiler', '-Wno-unused-variable',
]

pingpong_link_args = [
    '-gencode=arch=compute_89,code=sm_89',
]

# NVCC 12.9 workaround (same bug as in ep/meson.build)
nvcc = meson.get_compiler('cuda')
if nvcc.version().version_compare('>=12.9') and nvcc.version().version_compare('<12.10')
    pingpong_cpp_args  += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
    pingpong_cuda_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
endif

executable('nixl_device_pingpong',
    pingpong_sources,
    dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep, gpu_device_api_deps],
    include_directories: pingpong_inc_dirs,
    cpp_args: pingpong_cpp_args + gpu_device_api_cpp_args,
    cuda_args: pingpong_cuda_args + gpu_device_api_cuda_args,
    link_args: pingpong_link_args,
    link_with: gpu_device_api_link_with,
    build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
    install: false,
)

# Minimal "trust-by-inspection" variant: times only the nixlPut/nixlAtomicAdd
# call (issue) and the full RTT (one-way = RTT/2). No completion/stage polling.
simple_sources = [
    'simple_main.cpp',
    'bench_host.cpp',
    'simple_kernel.cu',
]

executable('nixl_device_pingpong_simple',
    simple_sources,
    dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep, gpu_device_api_deps],
    include_directories: pingpong_inc_dirs,
    cpp_args: pingpong_cpp_args + gpu_device_api_cpp_args,
    cuda_args: pingpong_cuda_args + gpu_device_api_cuda_args,
    link_args: pingpong_link_args,
    link_with: gpu_device_api_link_with,
    build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
    install: false,
)

# Windowed one-directional streaming BANDWIDTH bench (companion to the latency
# pingpong above). Reuses bench_host.cpp for setup; only the kernel/main differ.
stream_sources = [
    'stream_main.cpp',
    'bench_host.cpp',
    'stream_kernel.cu',
]

executable('nixl_device_bw_stream',
    stream_sources,
    dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep, gpu_device_api_deps],
    include_directories: pingpong_inc_dirs,
    cpp_args: pingpong_cpp_args + gpu_device_api_cpp_args,
    cuda_args: pingpong_cuda_args + gpu_device_api_cuda_args,
    link_args: pingpong_link_args,
    link_with: gpu_device_api_link_with,
    build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
    install: false,
)

if gpu_device_api_is_ucx
    # CPU-proxy comparison variant for the profiling scripts. The default
    # nixl_device_pingpong target above follows gpu_device_api_backend.
    pingpong_proxy_define = ['-DNIXL_GPU_DEVICE_BACKEND_PROXY']

    pingpong_proxy_inc_dirs = [
        nixl_inc_dirs,
        nixl_gpu_proxy_inc_dirs,
        include_directories('.'),
    ]

    executable('nixl_device_pingpong_proxy',
        pingpong_sources,
        dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep],
        include_directories: pingpong_proxy_inc_dirs,
        cpp_args: pingpong_cpp_args + pingpong_proxy_define,
        cuda_args: pingpong_cuda_args + pingpong_proxy_define,
        link_args: pingpong_link_args,
        link_with: [nixl_gpu_proxy_lib],
        build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
        install: false,
    )

    executable('nixl_device_pingpong_simple_proxy',
        simple_sources,
        dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep],
        include_directories: pingpong_proxy_inc_dirs,
        cpp_args: pingpong_cpp_args + pingpong_proxy_define,
        cuda_args: pingpong_cuda_args + pingpong_proxy_define,
        link_args: pingpong_link_args,
        link_with: [nixl_gpu_proxy_lib],
        build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
        install: false,
    )

    executable('nixl_device_bw_stream_proxy',
        stream_sources,
        dependencies: [nixl_dep, nixl_infra, cuda_dep, ucx_dep],
        include_directories: pingpong_proxy_inc_dirs,
        cpp_args: pingpong_cpp_args + pingpong_proxy_define,
        cuda_args: pingpong_cuda_args + pingpong_proxy_define,
        link_args: pingpong_link_args,
        link_with: [nixl_gpu_proxy_lib],
        build_rpath: nixl_lib_dir + ':' + nixl_infra_dir,
        install: false,
    )
endif
