export CUDA_VISIBLE_DEVICES=2,3,4,5
export DATA_DIR='data/nq_search'

WAND_PROJECT='Router-R1-Official'

#export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
#export EXPERIMENT_NAME=nh-bs64-ppo-llama3.2-3b-it-em
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nh-bs64-ppo-qwen2.5-3b-it-em
#export BASE_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues


#data.train_files=$DATA_DIR/train_nh_llama.parquet \
#data.val_files=$DATA_DIR/test_nh_llama.parquet \

# Attention: DataLoader is set to drop_last=True by default, please set data.val_batch_size to a reasonable value.

PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_nh_qwen.parquet \
    data.val_files=$DATA_DIR/test_nh_qwen.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=100 \
    trainer.total_training_steps=225 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    +reward_metric="em" \
    +cost_coe=0.0 \
    +api_base="http://127.0.0.1:8001/v1" \
    +api_key=["key"] \
    2>&1 | tee $EXPERIMENT_NAME.log
