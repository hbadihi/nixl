#!/bin/bash
# usage: run.sh [gpu] [arch] -- build fence.cu and run the standard sweep on one idle GPU, bound to the GPU's NUMA node
set -e
cd "$(dirname "$0")"
GPU=${1:-0}
ARCH=${2:-sm_90}
BDF=$(nvidia-smi -i $GPU --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f' | sed 's/^0000//')
NODE=$(cat /sys/bus/pci/devices/$BDF/numa_node)
CPUS=${CPUS:-$([ "$NODE" -ge 0 ] && cat /sys/devices/system/node/node$NODE/cpulist || echo 0-$(($(nproc) - 1)))}
${NVCC:-nvcc} -O2 -arch=$ARCH -o fence fence.cu
echo "# $(hostname): $(lscpu | sed -n 's/^Model name: *//p' | head -1); GPU $GPU: $(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader); NUMA node $NODE, cpus $CPUS"
run() { CUDA_VISIBLE_DEVICES=$GPU taskset -c $CPUS ./fence "$@"; }
for w in 1 8 64 256 768; do
    for m in 0 1 2 3 4; do
        run $w $m 7168
    done
done
for w in 1 64 256; do
    for m in 0 3; do
        run $w $m 0
    done
done
