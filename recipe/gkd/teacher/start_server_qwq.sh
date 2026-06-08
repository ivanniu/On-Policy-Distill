# ============================================================================
# QwQ-32B Teacher Server Launcher
# ----------------------------------------------------------------------------
# Differences vs. start_server.sh (Qwen3-8B version):
#   1) CKPT_PATH         : Qwen3-8B                 -> QwQ-32B
#   2) CUDA_VISIBLE_DEVS : 0,1,2,3 (4 cards)        -> 0,1,2,3,4,5,6,7 (8 cards)
#   3) --tp-size          : 4                        -> 8
#      Reason: QwQ-32B weights are ~65 GB bf16. TP=8 shards weights to
#      ~8 GB/card, leaving plenty of KV-cache room on H20 (96 GB/card).
#   4) Tunable vLLM engine knobs exposed as CLI flags (see variables below).
#      Previously hard-coded in vllm_engine.py; now passed via worker.py.
# ============================================================================

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH="QwQ-32B"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ------------- vLLM engine knobs (edit here, no need to touch Python) -------
TP_SIZE=8
# GPU memory vLLM can use. On H20 (96 GB), 0.85 is safe. Default was 0.5.
GPU_MEMORY_UTILIZATION=0.65
# Max tokens processed in one scheduling step (affects prefill batching).
MAX_NUM_BATCHED_TOKENS=8192
# Max prompt+generation length. QwQ supports 131072; cap lower to save KV.
# Training uses max_prompt=1024 + max_response=16384 = 17408, so 30720 is
# fine with headroom; bump if you see truncation warnings in training log.
MAX_MODEL_LEN=30720
# Number of top logprobs per token to return. Current OPD uses top-1 only.
N_LOGPROBS=1
# ----------------------------------------------------------------------------

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

# ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
# ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py \
    --backend $BACKEND \
    --tp-size $TP_SIZE \
    --n-logprobs $N_LOGPROBS \
    --ckpt-path $CKPT_PATH \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"
