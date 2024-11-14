#!/bin/bash
export HF_TOKEN='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'
export WANDB_API_KEY='327e1d6f7292f4e0626960a6116029641b4571fc'

NUM_NODES=1
DEVICES=8
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=$(($NUM_NODES * $DEVICES * $MICRO_BATCH_SIZE))

python ./examples/nlp/gpt/train_reward_model.py \
      trainer.num_nodes=$NUM_NODES \
      trainer.devices=$DEVICES \
      trainer.rm.max_epochs=3 \
      ++model.micro_batch_size=$MICRO_BATCH_SIZE \
      ++model.data.data_impl=jsonl \
      pretrained_checkpoint.restore_from_path=/fsx-Training/home/ronzhi/Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-Instruct.nemo \
      "model.data.data_prefix={train: ['./data/hs2/train_reg.jsonl'], validation: ['./data/hs2/val_reg.jsonl'], test: ['./data/hs2/val_reg.jsonl']}" \
      exp_manager.explicit_log_dir=/fsx-Training/home/ronzhi/west_results_model/1025_west_2a_1node_hs2_20k_3epoch_lr3e6 \
      trainer.rm.save_interval=300 \
      trainer.rm.val_check_interval=300 \
      exp_manager.create_wandb_logger=True \
      exp_manager.wandb_logger_kwargs.project=batch_rm \
      exp_manager.wandb_logger_kwargs.name=1025_west_2a_1node_hs2_20k_3epoch_lr3e6 \
      trainer.rm.max_steps=10000\
      ++model.tensor_model_parallel_size=8 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.activations_checkpoint_granularity="selective" \
      ++model.activations_checkpoint_method="uniform" \
      model.global_batch_size=$GLOBAL_BATCH_SIZE \
      model.optim.sched.constant_steps=200 \
      model.reward_model_type="regression" \
      model.regression.num_attributes=9