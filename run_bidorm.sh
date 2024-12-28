#!/bin/bash
export HF_TOKEN='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'
export WANDB_API_KEY='327e1d6f7292f4e0626960a6116029641b4571fc'

NUM_NODES=1
DEVICES=4
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(($NUM_NODES * $DEVICES * $MICRO_BATCH_SIZE))

# /opt/model/24-11-04_Llama-3.2-3B-it/model.nemo

# /opt/model/results_model/1223_3b_dorm_mix_80k_12label_2epoch_lr3e6_bs20/checkpoints/megatron_gpt.nemo


python ./examples/nlp/gpt/bidorm_3b_train.py \
      trainer.num_nodes=$NUM_NODES \
      trainer.devices=$DEVICES \
      trainer.rm.max_epochs=2 \
      ++model.micro_batch_size=$MICRO_BATCH_SIZE \
      ++model.data.data_impl=jsonl \
      pretrained_checkpoint.restore_from_path=/opt/model/24-11-04_Llama-3.2-3B-it/model.nemo \
      "model.data.data_prefix={train: ['./data/dorm-mix/dorm-train-80k-with-source.jsonl'], validation: ['./data/dorm-mix/dorm-test-5k.jsonl'], test: ['./data/dorm-mix/dorm-test-5k.jsonl']}" \
      exp_manager.explicit_log_dir=/opt/model/results_model/1227_3b_bidorm_partition_5_bs4_bilevel_debug \
      trainer.rm.save_interval=500 \
      trainer.rm.val_check_interval=500 \
      exp_manager.create_wandb_logger=True \
      exp_manager.wandb_logger_kwargs.project=dorm3b \
      exp_manager.wandb_logger_kwargs.name=1227_3b_bidorm_partition_5_bs4_bilevel_debug  \
      trainer.rm.max_steps=100000\
      ++model.tensor_model_parallel_size=4 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.activations_checkpoint_granularity="selective" \
      ++model.activations_checkpoint_method="uniform" \
      model.global_batch_size=$GLOBAL_BATCH_SIZE \
      model.optim.sched.constant_steps=500 \
      model.optim.sched.warmup_steps=10 \
      model.reward_model_type="regression" \
      model.regression.num_attributes=12 \
      model.use_weighted_training=True