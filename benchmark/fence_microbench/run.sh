#!/bin/bash
# usage: run.sh [gpu] [arch] -- build fence.cu and run the standard sweep on one idle GPU
set -e
cd "$(dirname "$0")"
GPU=${1:-0}
ARCH=${2:-sm_90}
${NVCC:-nvcc} -O2 -arch=$ARCH -o fence fence.cu
echo "# $(hostname): $(lscpu | sed -n 's/^Model name: *//p' | head -1); GPU $GPU: $(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader)"
for w in 1 8 64 256 768; do
    for m in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=$GPU ./fence $w $m 7168
    done
done
for w in 1 64 256; do
    for m in 0 3; do
        CUDA_VISIBLE_DEVICES=$GPU ./fence $w $m 0
    done
done
