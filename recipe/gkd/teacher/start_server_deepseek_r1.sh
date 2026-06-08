#!/usr/bin/env bash
# ============================================================================
# DeepSeek-R1-0528 Teacher Server Launcher (2-node, PP=2 x TP=8 + EP)
# ============================================================================
#
# Architecture: DeepSeek-R1 is a 671B MoE model (256 routed experts, 8 active).
# FP8 weights ~671 GB. Requires 2 nodes x 8 GPUs = 16 GPUs total.
#
# Deployment strategy:
#   - PP=2 across 2 nodes (minimal cross-node communication)
#   - TP=8 within each node (fast intra-node NVLink)
#   - EP=2 (expert parallel across nodes)
#   - vLLM v0.19 + Ray
#   - NCCL over IB/RoCE RDMA (8x mlx5_bond, 200Gb HDR, GDRDMA bypasses memlock limit)
#
# Node layout:
#   - Launcher (this node): ts-8b1d81139dfc7474019e0bd5685d0e90-launcher
#   - Worker node:          ts-8b1d81139dfc7474019e0bd5685d0e90-worker-0
#
# Usage:
#   Run this script on the launcher node.
#
# ============================================================================

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH="DeepSeek-R1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Disable HTTP proxy for this session — .bashrc sets http_proxy to star-proxy.oa.com:3128
# which causes Ray Compiled DAG gRPC connections between nodes (29.x.x.x) to be routed
# through the OA proxy, hanging all cross-node PP communication on first request.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*"

# NCCL configuration for cross-node communication
# IB/RoCE with GDRDMA: 8x 200Gb HDR bonds, ~30 GB/s cross-node bandwidth.
# GDRDMA pins GPU memory directly, bypassing the 64KB CPU memlock limit.
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=TRACE
export UCX_NET_DEVICES="bond1"

# Ensure nvcc is in PATH — flashinfer CUTLASS MoE backend needs JIT compilation
export PATH=/usr/local/cuda/bin:$PATH

# Ray Compiled DAG timeout — PP=2 first step can be slow during warmup
export RAY_CGRAPH_get_timeout=600

# # Force Ray Compiled DAG to use shared memory instead of NCCL for inter-PP tensor transport.
# # With "auto" (default), Ray tries NCCL which inits a NEW communicator using Ray's node IP
# # (bond8: 29.136.203.x), but NCCL_SOCKET_IFNAME=bond1 forces bootstrap to bond1 (29.226.2.x).
# # This IP mismatch causes the NCCL init to deadlock on the first inference request.
# # Setting "shm" avoids this; the actual model TP/EP NCCL still uses IB via torch.distributed.
# export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm

export VLLM_LOGGING_LEVEL=DEBUG

# Bypass Ray Compiled DAG entirely for PP execution — the Compiled DAG setup hangs
# on this cluster. Use serial PP execution via plain ray.remote() calls instead.
# NOTE: Root cause of DAG *compilation* deadlock identified and fixed (missing
# _ray_system concurrency group on actors). However, DAG *execution* still hangs
# due to SHM channel communication issue between EngineCore and PP0 workers.
# Keep serial PP fallback until the channel issue is also resolved.
export VLLM_DISABLE_RAY_COMPILED_DAG=1

export NCCL_P2P_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_DEBUG_SUBSYS=NET 


# ========================= Multi-node configuration =========================
LAUNCHER_NODE="ts-8b1d81139dfc7474019e0bd5685d0e90-launcher"
WORKER_NODE="ts-8b1d81139dfc7474019e0bd5685d0e90-worker-0"

RAY_HEAD_PORT=6399
RAY_DASHBOARD_PORT=8265

# ========================= vLLM engine knobs ================================
# PP=2: pipeline parallel across 2 nodes (only activation tensors cross-node)
# TP=8: tensor parallel within each node (NVLink, no cross-node)
# EP=2: expert parallel across 2 nodes
PP_SIZE=2
TP_SIZE=8
EP_SIZE=2
GPU_MEMORY_UTILIZATION=0.70
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=30720
N_LOGPROBS=1
DISTRIBUTED_BACKEND="ray"
# ============================================================================

wait_server_ready() {
    local server="$1"
    local ip="$2"
    local port="$3"
    while true; do
        echo "wait $server server ready at $ip:$port..."
        if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$ip',$port)); s.close()" 2>/dev/null; then
            break
        fi
        sleep 2
    done
}

# ============================================================================
# Step 1: Start Ray cluster
# ============================================================================
echo "============================================"
echo " Step 1: Starting Ray cluster"
echo "============================================"

ray stop --force 2>/dev/null
ssh "$WORKER_NODE" "ray stop --force" 2>/dev/null
sleep 2

# Pass NCCL env vars via Ray runtime_env so all workers inherit them
export RAY_OVERRIDE_JOB_RUNTIME_ENV='{"env_vars":{"NCCL_IB_DISABLE":"0","NCCL_IB_HCA":"mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8","NCCL_IB_GID_INDEX":"3","NCCL_IB_TC":"160","NCCL_IB_TIMEOUT":"22","NCCL_IB_SL":"3","NCCL_SOCKET_IFNAME":"bond1","NCCL_DEBUG":"WARN","RAY_CGRAPH_get_timeout":"600","VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":"shm","PATH":"/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","http_proxy":"","https_proxy":"","HTTP_PROXY":"","HTTPS_PROXY":"","no_proxy":"*"}}'

ray start --head --port=$RAY_HEAD_PORT --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus=8
echo "Ray head started on $LAUNCHER_NODE:$RAY_HEAD_PORT"

LAUNCHER_IP=$(hostname -I | awk '{print $1}')
RAY_HEAD_ADDR="${LAUNCHER_IP}:${RAY_HEAD_PORT}"
echo "Ray head address: $RAY_HEAD_ADDR"

ssh "$WORKER_NODE" "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export no_proxy='*' && export PATH=/usr/local/cuda/bin:\$PATH && export NCCL_IB_DISABLE=0 && export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8 && export NCCL_IB_GID_INDEX=3 && export NCCL_IB_TC=160 && export NCCL_IB_TIMEOUT=22 && export NCCL_IB_SL=3 && export NCCL_SOCKET_IFNAME=bond1 && export NCCL_DEBUG=WARN && export RAY_CGRAPH_get_timeout=600 && ray start --address='$RAY_HEAD_ADDR' --num-gpus=8"
echo "Ray worker started on $WORKER_NODE"

echo "Waiting for Ray cluster to have 16 GPUs..."
for i in $(seq 1 60); do
    N_GPUS=$(python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
print(int(ray.cluster_resources().get('GPU', 0)))
ray.shutdown()
" 2>/dev/null) || N_GPUS="0"
    if [ "${N_GPUS:-0}" -ge 16 ] 2>/dev/null; then
        echo "Ray cluster ready: $N_GPUS GPUs available"
        break
    fi
    echo "  ... ${N_GPUS:-0} GPUs so far, waiting..."
    sleep 5
done

# ============================================================================
# Step 2: Start ZMQ proxy
# ============================================================================
echo ""
echo "============================================"
echo " Step 2: Starting ZMQ proxy"
echo "============================================"

# Kill old processes (ignore errors if none found)
pkill -f "python proxy.py" 2>/dev/null || true
pkill -f "python worker.py" 2>/dev/null || true
sleep 1

nohup python proxy.py &> proxy.log &
echo "Proxy PID: $!"

wait_server_ready proxy localhost $PROXY_BACKEND_PORT
echo "ZMQ proxy is ready"

# ============================================================================
# Step 3: Start vLLM worker (PP=2 x TP=8 + EP, Ray across 2 nodes)
# ============================================================================
echo ""
echo "============================================"
echo " Step 3: Starting vLLM worker"
echo " PP=$PP_SIZE, TP=$TP_SIZE, EP=$EP_SIZE, backend=$DISTRIBUTED_BACKEND"
echo "============================================"

nohup python worker.py \
    --backend $BACKEND \
    --tp-size $TP_SIZE \
    --pp-size $PP_SIZE \
    --ep-size $EP_SIZE \
    --n-logprobs $N_LOGPROBS \
    --ckpt-path $CKPT_PATH \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    --distributed-executor-backend $DISTRIBUTED_BACKEND \
    &> worker.log &
echo "Worker PID: $!"

echo ""
echo "============================================"
echo " Teacher server launching..."
echo " Proxy: $LAUNCHER_NODE:$PROXY_FRONTEND_PORT"
echo " Monitor: tail -f worker.log"
echo " (Model loading may take 5-10 minutes)"
echo "============================================"
