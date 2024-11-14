#!/bin/bash

export HF_TOKEN='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'
export WANDB_API_KEY='327e1d6f7292f4e0626960a6116029641b4571fc'

python ./examples/nlp/gpt/train_rm_12b.py \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      trainer.rm.max_epochs=3 \
      ++model.micro_batch_size=4 \
      ++model.data.data_impl=jsonl \
      pretrained_checkpoint.restore_from_path=/workspace/Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-Instruct.nemo \
      "model.data.data_prefix={train: ['./data/hs2/train_reg.jsonl'], validation: ['./data/hs2/val_reg.jsonl'], test: ['./data/hs2/val_reg.jsonl']}" \
      exp_manager.explicit_log_dir=/results/20241022_12b_hs2  \
      trainer.rm.save_interval=300 \
      trainer.rm.val_check_interval=300 \
      exp_manager.create_wandb_logger=True \
      exp_manager.wandb_logger_kwargs.project=12b \
      exp_manager.wandb_logger_kwargs.name=20241022_12b_hs2 \
      trainer.rm.max_steps=10000\
      ++model.tensor_model_parallel_size=8 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.activations_checkpoint_granularity="selective" \
      ++model.activations_checkpoint_method="uniform" \
      model.global_batch_size=32 \
      model.optim.sched.constant_steps=500 \
      model.reward_model_type="regression" \
      model.regression.num_attributes=9