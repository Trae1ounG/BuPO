#!/usr/bin/env bash
set -xeuo pipefail
# export WANDB_MODE=online
# export WANDB_BASE_URL=YOUR_WANDB_BASE_URL
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
project_name='BuPO'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))

loss_agg_mode="token-mean"
n_resp_per_prompt=8

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
validation_data_dir=${validation_data_dir:-"${PWD}/validation_data"}
MODEL_PATH=${MODEL_PATH:-"your/model/path"}
DATA_PATH=${DATA_PATH:-"your/data/path"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_PATH}/deepmath-5k.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_PATH}/aime_2024.parquet","${DATA_PATH}/aime_2025.parquet","${DATA_PATH}/amc2023.parquet","${DATA_PATH}/math500.parquet"]}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 32))
infer_ppo_max_token_len=$((1024 * 32))
offload=True
gen_tp=2
k=5
iterative_steps=30
prompt_template_type="qwen3_no_thinking"
experiment_name="modelname_bupo_deepmath5k_${k}layerpolicy_iterstep${iterative_steps}_bsz128_mini32_n8_resp8k_higherclip0.2_lr1e-6"

# ray job submit --runtime-env=verl/trainer/runtime_env.yaml --no-wait --
python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.prompt_template_type=$prompt_template_type \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.internal_policy_interative=True \
    actor_rollout_ref.actor.internal_layer=${k} \
    actor_rollout_ref.actor.iterative_steps=${iterative_steps} \
    actor_rollout_ref.model.override_config.internal_layer=${k} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=30 \
    trainer.save_freq=100 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir=${PWD}/checkpoints/${project_name}/${experiment_name} \
    trainer.validation_data_dir=${PWD}/valid_data/${project_name}/${experiment_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.resume_mode=auto  $@