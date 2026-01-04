set -x

export PYTHONPATH=`pwd`:$PYTHONPATH
nnodes=1
tp_size=2
# Vllm
n_samples=1
temperature=1
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))

# Path
base_dir=MODEL_PATH
model=Qwen3-4B
model_path=${base_dir}/${model}
output_dir=BUPO_PATH/output_gen/${model}
data_dir=BUPO_PATH/data
dataset=math500

python -m verl.trainer.main_generation \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=8 \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.parquet \
    data.output_path=${output_dir}/${dataset}/${dataset}_test.parquet \
    data.batch_size=8 \
    data.n_samples=${n_samples} \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.8 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.tensor_model_parallel_size=${tp_size} \
    rollout.temperature=$temperature \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=$max_prompt_length \
    rollout.response_length=$max_response_length \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))