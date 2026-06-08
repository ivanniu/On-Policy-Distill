#!/usr/bin/env bash
set -xeuo pipefail

project_name='ON_POLICY_DISTILL'
exp_name='OPD'

adv_estimator=opd

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Clamp per-token advantage to [-opd_loss_max_clamp, +opd_loss_max_clamp].
# Prevents extreme teacher-student logp gaps (e.g. -30) from dominating the gradient.
opd_loss_max_clamp=10.0

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 16))

loss_agg_mode="token-mean"

train_prompt_bsz=128
n_resp_per_prompt=1
train_prompt_mini_bsz=128

# Ray cluster
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths — modify these to match your environment
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}/data"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/student"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/val.parquet"}

# Teacher server configuration
export TEACHER_SERVER_IP=${TEACHER_SERVER_IP:-"127.0.0.1"}
export TEACHER_SERVER_PORT=${TEACHER_SERVER_PORT:-"15555"}
export TEACHER_N_WORKERS=${TEACHER_N_WORKERS:-"1"}
export TEACHER_CKPT_PATH=${TEACHER_CKPT_PATH:-"${RAY_DATA_HOME}/models/teacher"}
export TEACHER_MAX_SEQ_LEN=${TEACHER_MAX_SEQ_LEN:-"30720"}
export HYDRA_FULL_ERROR=1

# Sampling parameters
temperature=1.0
top_p=1.0
top_k=-1  # -1 for vLLM rollout

val_top_p=0.95
val_top_k=20
val_temperature=0.6

# Performance parameters
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=2
fsdp_size=8

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.shuffle=True \
    data.seed=44 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.opd_loss_max_clamp=${opd_loss_max_clamp} \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.policy_loss.loss_mode="opd" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.nccl_timeout=72000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=31744 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=opd \
    reward_model.enable=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=25 \
    trainer.save_freq=500 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
