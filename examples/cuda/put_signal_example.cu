/* 
Compile command :

nvcc -arch=sm_90\
  -I$NIXL_HOME/include \
  -I/workspace/nixl/include/gpu/ucx \
  -I$UCX_HOME/include \
  -L$NIXL_HOME/lib/x86_64-linux-gnu \
  -L/usr/lib/x86_64-linux-gnu \
  -L/workspace/nixl/build/subprojects/abseil-cpp-20240722.0 \
  -Wno-deprecated-gpu-targets \
  put_signal_example.cu \
  -lnixl -lnixl_build -lnixl_common -lserdes -lstream \
  -labsl_log \
  -labsl_synchronization \
  -labsl_strings \
  -labsl_status \
  -labsl_debugging \
  -labsl_time \
  -labsl_base \
  -Xlinker -rpath=$NIXL_HOME/lib/x86_64-linux-gnu \
  -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE \
  -o put_signal_example

Run command (example for two GPUs on localhost):

# On GPU 0 (listens on port 5555, connects to GPU 1 on port 5556)
UCX_WARN_UNUSED_ENV_VARS=n LD_PRELOAD=$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_common.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_gpunetio.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_verbs.so \
LD_LIBRARY_PATH=$NIXL_HOME/lib/x86_64-linux-gnu:$UCX_HOME/lib:$LD_LIBRARY_PATH \
./put_signal_example 0 127.0.0.1 5556

# On GPU 1 (listens on port 5556, connects to GPU 0 on port 5555)
UCX_WARN_UNUSED_ENV_VARS=n LD_PRELOAD=$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_common.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_gpunetio.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_verbs.so \
LD_LIBRARY_PATH=$NIXL_HOME/lib/x86_64-linux-gnu:$UCX_HOME/lib:$LD_LIBRARY_PATH \
./put_signal_example 1 127.0.0.1 5555

*/

#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include "nixl.h"
#include "nixl_types.h"
#include "nixl_device.cuh"

#define INITIATOR 0
#define TARGET 1

// ====================================================================================
// Constants
// ====================================================================================

constexpr size_t BUFFER_SIZE = 1024;
constexpr int MAX_METADATA_POLL_ATTEMPTS = 150;
constexpr int METADATA_POLL_INTERVAL_MS = 200;
constexpr int PEER_INIT_WAIT_SECONDS = 3;
constexpr int BACKGROUND_THREAD_WAIT_SECONDS = 2;
constexpr int NOTIF_POLL_INTERVAL_SECONDS = 1;

enum class FillPattern : unsigned char {
    INITIATOR_DATA = 0xaa,
    TARGET_EMPTY = 0x00
};

struct BufferInfo {
    void* addr;
    size_t len;
};

__global__ void InitiatorKernel(void* data_addr, nixlGpuXferReqH  req_handle)
{
    __shared__ nixlGpuXferStatusH xfer_status;
    nixl_status_t status = nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::THREAD>(
        req_handle,     // The request handle
        0,              // The index of the descriptor to use (we have one)
        0,              // The local offset to start from
        0,              // The remote offset to write to
        BUFFER_SIZE,           // The size of the transfer
        0,              // Default channel ID
        true,           // Default no_delay
        &xfer_status    // The status handle
    );

    if (status < NIXL_SUCCESS) {
        printf("InitiatorKernel: nixlGpuPostSingleWriteXferReq failed: status=%d\n", status);
        return;
    }

    while (status != NIXL_SUCCESS) {
        status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(xfer_status);
    }   

    // Send signal after data transfer completes (descriptor index 1)
    status = nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
        req_handle,
        1,              // descriptor index 1 (signal buffer)
        1,              // signal_inc = 1 (increment by 1)
        0,              // signal_offset
        0,              // channel_id
        true,           // no_delay
        &xfer_status
    );
    if (status < NIXL_SUCCESS) {
        printf("InitiatorKernel: nixlGpuPostSignalXferReq failed: status=%d\n", status);
        return;
    }

    while (status != NIXL_SUCCESS) {
        status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(xfer_status);
    }
    printf("InitiatorKernel: signal sent successfully\n");
}

__global__ void TargetKernel(void* data_addr, void* signal_addr)
{
    while(nixlGpuReadSignal<nixl_gpu_level_t::THREAD>(signal_addr) == 0) {
        ;
    }
    volatile unsigned char* bytes = (volatile unsigned char*)data_addr;
    assert(bytes[0] == static_cast<unsigned char>(FillPattern::INITIATOR_DATA));
    printf("TargetKernel: data received successfully\n");

}

void nixl_exit_on_failure(nixl_status_t status, std::string_view message, std::string_view agent = "") {
    if (status == NIXL_SUCCESS) return;
    std::cout << "ERROR: " << message << (agent.empty() ? "" : " for agent " + std::string{agent}) << ": "
              << nixlEnumStrings::statusStr(status) << " [" << status << "]" << std::endl;
    exit(EXIT_FAILURE);
}

std::string get_remote_agent_name(int device_id) {
    return (device_id == INITIATOR) ? "agent_1" : "agent_0";
}

int get_remote_port(int device_id) {
    return (device_id == INITIATOR) ? 5556 : 5555;
}

struct CliArgs {
    int device_id;
    std::string remote_ip;
    int remote_port;
};

CliArgs parse_arguments(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <device_id> <remote_ip> <remote_port>\n";
        std::cerr << "Example: " << argv[0] << " 0 127.0.0.1 5556\n";
        exit(EXIT_FAILURE);
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount < 2) {
        std::cerr << "Error: Need at least 2 GPUs, detected " << deviceCount << "\n";
        exit(EXIT_FAILURE);
    }

    CliArgs args;
    args.device_id = atoi(argv[1]);
    args.remote_ip = argv[2];
    args.remote_port = atoi(argv[3]);
    
    cudaSetDevice(args.device_id);
    return args;
}

std::string get_agent_name(int device_id) {
    return "agent_" + std::to_string(device_id);
}

std::unique_ptr<nixlAgent> initialize_nixl_agent(int device_id) {
    std::string agent_name = get_agent_name(device_id);
    bool is_listener = true;
    int port = (device_id == INITIATOR) ? 5555 : 5556;
    nixlAgentConfig cfg(true, is_listener, port);
    return std::make_unique<nixlAgent>(agent_name, cfg);
}

void setup_backend(nixlAgent& agent,const std::string& agent_name, nixlBackendH*& backend) {
    const std::string backend_type = "UCX";
    nixl_mem_list_t mems;
    nixl_b_params_t params;

    nixl_status_t ret = agent.createBackend(backend_type, params, backend);
    nixl_exit_on_failure(ret, "createBackend", agent_name);
}

void allocate_and_register_memory(nixlAgent& agent, int device_id, BufferInfo& data_info, size_t data_len, BufferInfo& signal_info, size_t signal_len, nixlBackendH* backend) {
    data_info.len = data_len;
    signal_info.len = signal_len;
    cudaError_t data_err = cudaMalloc(&data_info.addr, data_len);
    cudaError_t signal_err = cudaMalloc(&signal_info.addr, signal_len);
    if (data_err != cudaSuccess || signal_err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(data_err) << std::endl;
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(signal_err) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaMemset(signal_info.addr, 0, signal_info.len);
    cudaMemset(data_info.addr, (device_id == INITIATOR) ? 0xaa : 0x00, data_info.len);


    nixlBlobDesc data_buff;
    data_buff.addr = (uintptr_t)data_info.addr;
    data_buff.len = data_info.len;
    data_buff.devId = device_id;

    nixlBlobDesc signal_buff;
    signal_buff.addr = (uintptr_t)signal_info.addr;
    signal_buff.len = signal_info.len;
    signal_buff.devId = device_id;

    nixl_reg_dlist_t data_dlist(VRAM_SEG);
    nixl_reg_dlist_t signal_desc_list(VRAM_SEG);

    data_dlist.addDesc(data_buff);
    signal_desc_list.addDesc(signal_buff);

    nixl_status_t ret = agent.registerMem(data_dlist);
    nixl_exit_on_failure(ret, "registerMem");

    ret = agent.registerMem(signal_desc_list);
    nixl_exit_on_failure(ret, "registerMem");

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);
    ret = agent.prepGpuSignal(signal_desc_list, &extra_params);
    nixl_exit_on_failure(ret, "prepGpuSignal");
}

bool wait_for_remote_metadata(nixlAgent& agent, const std::string& remote_agent_name) {
    nixl_xfer_dlist_t check_descs(VRAM_SEG);
    
    for (int attempt = 0; attempt < MAX_METADATA_POLL_ATTEMPTS; ++attempt) {
        if (agent.checkRemoteMD(remote_agent_name, check_descs) == NIXL_SUCCESS) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(METADATA_POLL_INTERVAL_MS));
    }
    return false;
}

void exchange_metadata(nixlAgent& agent, int device_id, const std::string& remote_ip, int remote_port) {
    std::string remote_agent_name = get_remote_agent_name(device_id);
    nixl_opt_args_t extra_params;
    extra_params.ipAddr = remote_ip;
    extra_params.port = remote_port;

    std::cout << "Waiting for peers to initialize..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(PEER_INIT_WAIT_SECONDS));

    std::cout << "Sending local metadata to " << remote_agent_name << " at " 
              << extra_params.ipAddr << ":" << extra_params.port << "..." << std::endl;
    nixl_status_t ret = agent.sendLocalMD(&extra_params);
    nixl_exit_on_failure(ret, "sendLocalMD");
    
    std::cout << "Polling for metadata from " << remote_agent_name << "..." << std::endl;
    if (!wait_for_remote_metadata(agent, remote_agent_name)) {
        nixl_exit_on_failure(NIXL_ERR_NOT_FOUND, "Timeout waiting for remote metadata");
    }

    ret = agent.fetchRemoteMD(remote_agent_name, &extra_params);
    nixl_exit_on_failure(ret, "fetchRemoteMD");
    std::cout << "✅ Successfully exchanged metadata with " << remote_agent_name << std::endl;
}

std::pair<BufferInfo, BufferInfo> exchange_buffer_info(nixlAgent& agent, int device_id, 
                                                        BufferInfo data_local_info, BufferInfo signal_local_info) {
    std::string remote_agent_name = get_remote_agent_name(device_id);
    
    // Combine both buffer info into one message: "data_addr:data_len;signal_addr:signal_len"
    std::string buffer_info = std::to_string((uintptr_t)data_local_info.addr) + ":" + std::to_string(data_local_info.len) +
                              ";" + std::to_string((uintptr_t)signal_local_info.addr) + ":" + std::to_string(signal_local_info.len);
    nixl_status_t ret = agent.genNotif(remote_agent_name, buffer_info);
    nixl_exit_on_failure(ret, "genNotif");

    nixl_notifs_t notif_map;
    // Wait for the notification to arrive
    while (notif_map.find(remote_agent_name) == notif_map.end() || notif_map[remote_agent_name].empty()) {
        std::this_thread::sleep_for(std::chrono::seconds(NOTIF_POLL_INTERVAL_SECONDS));
        ret = agent.getNotifs(notif_map);
        nixl_exit_on_failure(ret, "getNotifs");
    }
    
    // Parse remote info: "data_addr:data_len;signal_addr:signal_len"
    auto& info = notif_map[remote_agent_name][0];
    auto semicolon = info.find(';');
    
    // Parse data buffer
    std::string data_part = info.substr(0, semicolon);
    auto colon1 = data_part.find(':');
    BufferInfo data_remote = {
        (void*)std::stoull(data_part.substr(0, colon1)),
        std::stoull(data_part.substr(colon1 + 1))
    };
    
    // Parse signal buffer
    std::string signal_part = info.substr(semicolon + 1);
    auto colon2 = signal_part.find(':');
    BufferInfo signal_remote = {
        (void*)std::stoull(signal_part.substr(0, colon2)),
        std::stoull(signal_part.substr(colon2 + 1))
    };
    
    return {data_remote, signal_remote};
}

void perform_transfer(nixlAgent& agent, int device_id, BufferInfo data_local_info, BufferInfo data_remote_info, BufferInfo signal_remote_info,
     BufferInfo signal_local_info) {
    if (device_id == INITIATOR) {
        std::string remote_agent_name = get_remote_agent_name(device_id);

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        nixlBasicDesc data_local_desc = { (uintptr_t)data_local_info.addr, data_local_info.len, (uint32_t)device_id };
        local_descs.addDesc(data_local_desc);
        nixlBasicDesc signal_local_desc = { (uintptr_t)signal_local_info.addr, signal_local_info.len, (uint32_t)device_id };
        local_descs.addDesc(signal_local_desc);

        nixl_xfer_dlist_t remote_descs(VRAM_SEG);
        nixlBasicDesc data_remote_desc = { (uintptr_t)data_remote_info.addr, data_remote_info.len, TARGET };
        remote_descs.addDesc(data_remote_desc);
        nixlBasicDesc signal_remote_desc = { (uintptr_t)signal_remote_info.addr, signal_remote_info.len, TARGET };
        remote_descs.addDesc(signal_remote_desc);
        
        nixlXferReqH* req_handle;

        nixl_status_t ret = agent.createXferReq(NIXL_WRITE, local_descs, remote_descs, 
                                                remote_agent_name, req_handle);
        nixl_exit_on_failure(ret, "createXferReq");
        
        nixlGpuXferReqH gpu_req_handle;
        ret = agent.createGpuXferReq(*req_handle, gpu_req_handle);
        nixl_exit_on_failure(ret, "createGpuXferReq");
        
        InitiatorKernel<<<1, 1>>>(data_local_info.addr, gpu_req_handle);
        cudaDeviceSynchronize();
    } else { // TARGET
        std::cout << "TARGET: waiting for transfer in kernel..." << std::endl;
        TargetKernel<<<1, 1>>>(data_local_info.addr, signal_local_info.addr);
        cudaDeviceSynchronize();
    }
    std::cout << "Kernel launched and synchronized successfully" << std::endl;
}

void finalize_and_wait() {
    std::this_thread::sleep_for(std::chrono::seconds(BACKGROUND_THREAD_WAIT_SECONDS));
}

int main(int argc, char** argv) {
    CliArgs args = parse_arguments(argc, argv);
    
    std::unique_ptr<nixlAgent> agent = initialize_nixl_agent(args.device_id);
    
    nixlBackendH* backend;
    setup_backend(*agent, get_agent_name(args.device_id), backend);

    BufferInfo data_local_info, signal_local_info;
    allocate_and_register_memory(*agent, args.device_id, data_local_info, BUFFER_SIZE, signal_local_info, BUFFER_SIZE, backend);

    exchange_metadata(*agent, args.device_id, args.remote_ip, args.remote_port);
    
    auto [data_remote_info, signal_remote_info] = exchange_buffer_info(*agent, args.device_id, 
                                                           data_local_info,
                                                           signal_local_info);
    
    perform_transfer(*agent, args.device_id, data_local_info, 
                    data_remote_info, signal_remote_info, signal_local_info);

    finalize_and_wait();
    
    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}
