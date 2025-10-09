#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3,4,5
export DATA_DIR='data/nq_search'
export VERL_PPO_LOGGING_LEVEL=DEBUG

WAND_PROJECT='Router-R1-Official'

#export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
#export EXPERIMENT_NAME=nh-bs64-ppo-llama3.2-3b-it-em
export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
export EXPERIMENT_NAME=nh-bs64-ppo-qwen2.5-3b-it-em
#export BASE_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-key}}"

#data.train_files=$DATA_DIR/train_nh_qwen.parquet \
#data.val_files=$DATA_DIR/test_nh_qwen.parquet \

# Attention: DataLoader is set to drop_last=True by default, please set data.val_batch_size to a reasonable value.

find_latest_checkpoint() {

    local ckpt_dir="$1"
    local latest_path=""
    local latest_step=-1
    [[ -d "$ckpt_dir" ]] || return

    shopt -s nullglob
    for path in "$ckpt_dir"/global_step_*; do
        [[ -d "$path" ]] || continue
        local step=${path##*_}
        [[ $step =~ ^[0-9]+$ ]] || continue

        if compgen -G "$path"/pytorch_model*.bin > /dev/null ||
           compgen -G "$path"/*.safetensors > /dev/null ||
           [[ -f "$path/pytorch_model.bin.index.json" ]] ||
           [[ -f "$path/model.safetensors.index.json" ]]; then
            if (( step > latest_step )); then
                latest_step=$step
                latest_path=$path
            fi
        fi
    done
    shopt -u nullglob

    if [[ -n "$latest_path" ]]; then
        printf '%s' "$latest_path"
    fi

    return 0
}
CHECKPOINT_ROOT="verl_checkpoints/$EXPERIMENT_NAME"
ACTOR_CKPT_DIR="$CHECKPOINT_ROOT/actor"
CRITIC_CKPT_DIR="$CHECKPOINT_ROOT/critic"

mkdir -p "$ACTOR_CKPT_DIR" "$CRITIC_CKPT_DIR"

LOAD_ACTOR_CKPT="$(find_latest_checkpoint "$ACTOR_CKPT_DIR")"

LOAD_CRITIC_CKPT="$(find_latest_checkpoint "$CRITIC_CKPT_DIR")"

if [[ -z "$LOAD_ACTOR_CKPT" && -z "$LOAD_CRITIC_CKPT" ]]; then
    echo "No existing checkpoints found. Starting training from scratch." >&2
fi
HYDRA_ARGS=(
    "data.train_files=$DATA_DIR/train_nh_llama.parquet"
    "data.val_files=$DATA_DIR/test_nh_llama.parquet"
    "data.train_batch_size=16"
    "data.val_batch_size=16"
    "data.max_prompt_length=4096"
    "data.max_response_length=512"
    "data.max_start_length=4096"
    "data.max_obs_length=600"
    "data.filter_overlong_prompts=True"
    "data.truncation='error'"
    "data.shuffle_train_dataloader=True"
    "algorithm.adv_estimator=grpo"
    "actor_rollout_ref.model.path=$BASE_MODEL"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.model.lora_rank=64"
    "actor_rollout_ref.model.lora_alpha=32"
    "actor_rollout_ref.model.target_modules=all-linear"
    "actor_rollout_ref.actor.ppo_mini_batch_size=8"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.use_remove_padding=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.fsdp_config.grad_offload=true"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=16"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.ref.log_prob_micro_batch_size=16"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "actor_rollout_ref.rollout.n=2"
    "actor_rollout_ref.rollout.n_agent=1"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.actor.state_masking=true"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "algorithm.no_think_rl=false"
    "algorithm.use_kl_in_reward=False"
    "trainer.critic_warmup=0"
    "trainer.logger=['wandb']"
    "trainer.n_gpus_per_node=4"
    "trainer.nnodes=1"
    "trainer.save_freq=4"
    "trainer.test_freq=1"
    "trainer.project_name=$WAND_PROJECT"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.total_epochs=100"
    "trainer.total_training_steps=225"
    "trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME"
    "max_turns=4"
    "+reward_metric=em"
    "+cost_coe=0.0"
    "+api_base=https://openrouter.ai/api/v1"
    "+api_key=$API_KEY"
)

if [[ -n "$LOAD_ACTOR_CKPT" ]]; then
    echo "Auto-loading actor checkpoint from $LOAD_ACTOR_CKPT"
    HYDRA_ARGS+=("+actor_rollout_ref.resume_from_checkpoint=$LOAD_ACTOR_CKPT")
    HYDRA_ARGS+=("+trainer.resume_from_checkpoint=$LOAD_ACTOR_CKPT")
elif [[ -n "$LOAD_CRITIC_CKPT" ]]; then
    echo "Auto-loading trainer state from critic checkpoint $LOAD_CRITIC_CKPT"
    HYDRA_ARGS+=("+trainer.resume_from_checkpoint=$LOAD_CRITIC_CKPT")
fi

if [[ -n "$LOAD_CRITIC_CKPT" ]]; then
    echo "Auto-loading critic checkpoint from $LOAD_CRITIC_CKPT"
    HYDRA_ARGS+=("+critic.resume_from_checkpoint=$LOAD_CRITIC_CKPT")
fi

echo "Launching training with arguments:" >&2
printf '  %q\n' python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" >&2

PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" \
    2>&1 | tee "$EXPERIMENT_NAME.log"
