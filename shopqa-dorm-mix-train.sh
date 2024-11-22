#!/bin/bash
echo "Current working directory: $(pwd)"
echo "Listing current directory:"
ls -la
echo "Listing parent directories:"
ls -la ..
ls -la ../..

export HF_TOKEN='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'
export WANDB_API_KEY='327e1d6f7292f4e0626960a6116029641b4571fc'
export HYDRA_FULL_ERROR=1

EXPERIMENT_NAME="shopqa-dorm-mix-80k-12label"
DATE=$(date +%Y%m%d_%H%M)
NUM_NODES=2
DEVICES=8
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=$(($NUM_NODES * $DEVICES * $MICRO_BATCH_SIZE))
EPOCHS=3
EXPERIMENT_ID="${DATE}_${NUM_NODES}node_${EXPERIMENT_NAME}_${EPOCHS}epoch_lr3e6"
OUTPUT_DIR="./training_results/${EXPERIMENT_ID}"

mkdir -p $OUTPUT_DIR

#echo "Downloading files from S3..."
#aws s3 cp s3://shopqa-users/ronzhi/Mistral-NeMo-12B-Instruct ./ --recursive
#aws s3 cp s3://shopqa-users/ronzhi/12b ./ --recursive

#if [ ! -f "./Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-Instruct.nemo" ] || \
#  [ ! -f "./Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-tokenizer.json" ] || \
#  [ ! -f "./Mistral-NeMo-12B-Instruct/README.md" ] || \
#  [ ! -f "./data/dorm-mix/dorm-train-80k.jsonl" ] || \
#  [ ! -f "./data/dorm-mix/dorm-test-5k.jsonl" ]; then
#   echo "Error: Some files failed to download!"
#   exit 1
#fi

#echo "All required files downloaded successfully. Starting training..."

python ./examples/nlp/gpt/train_reward_model.py \
     trainer.num_nodes=$NUM_NODES \
     trainer.devices=8 \
     trainer.rm.max_epochs=$EPOCHS \
     ++model.micro_batch_size=$MICRO_BATCH_SIZE \
     ++model.data.data_impl=jsonl \
     pretrained_checkpoint.restore_from_path=../../Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-Instruct.nemo \
     "model.data.data_prefix={train: ['./data/dorm-mix/dorm-train-80k.jsonl'], validation: ['./data/dorm-mix/dorm-test-5k.jsonl'], test: ['./data/dorm-mix/dorm-test-5k.jsonl']}" \
     exp_manager.explicit_log_dir=$OUTPUT_DIR \
     trainer.rm.save_interval=300 \
     trainer.rm.val_check_interval=300 \
     exp_manager.create_wandb_logger=True \
     exp_manager.wandb_logger_kwargs.project=dorm \
     exp_manager.wandb_logger_kwargs.name=${EXPERIMENT_ID} \
     trainer.rm.max_steps=15000 \
     ++model.tensor_model_parallel_size=8 \
     ++model.pipeline_model_parallel_size=1 \
     ++model.activations_checkpoint_granularity="selective" \
     ++model.activations_checkpoint_method="uniform" \
     model.global_batch_size=$GLOBAL_BATCH_SIZE \
     model.optim.sched.constant_steps=200 \
     model.reward_model_type="regression" \
     model.regression.num_attributes=12

# 训练完成后上传结果
echo "Training completed. Uploading results to S3..."
if [ "${AWS_BATCH_JOB_NODE_INDEX:-0}" = "0" ]; then
    echo "Training completed. Main node uploading results to S3..."
    aws s3 cp $OUTPUT_DIR s3://shopqa-users/ronzhi/experiments/${EXPERIMENT_ID} --recursive
    echo "Results uploaded to: s3://shopqa-users/ronzhi/experiments/${EXPERIMENT_ID}"
else
    echo "Training completed on worker node ${AWS_BATCH_JOB_NODE_INDEX}"
fi
