#!/bin/bash

# tar -xvf .nemo

#SOURCE_CHECKPOINT_PATH="/workspace/12b/NeMo-Aligner/results_model/1119_ec2_1node_dorm_8k_2epoch_lr3e6/checkpoints"
#EXAMPLE_STRUCTURE_PATH="/workspace/NeMo-Aligner/results_model/rm_8b_inst_hs2/checkpoints/nemo_example"
#TARGET_CHECKPOINT_PATH="/workspace/NeMo-Aligner/results_model/oh_mix_oa2_hs2_0717/checkpoints/nemo_ckpt_oh_mix_large_diff_step_300_consumed_samples_24000"

SOURCE_CHECKPOINT_PATH="/opt/model/results_model/1227_3b_bidorm_partition_5_bs8/checkpoints/megatron_gpt--val_loss=0.612-step=14000-consumed_samples=112000-epoch=1"
EXAMPLE_STRUCTURE_PATH="/opt/model/results_model/1223_3b_dorm_mix_80k_12label_2epoch_lr3e6_bs20/checkpoints/3b_nemo_example"
TARGET_CHECKPOINT_PATH="/opt/model/results_model/1227_3b_bidorm_partition_5_bs8/checkpoints/bidorm_3b_ckpt_step_14000_consumed_samples_112000"


mkdir -p ${TARGET_CHECKPOINT_PATH}

cp ${EXAMPLE_STRUCTURE_PATH}/model_config.yaml ${TARGET_CHECKPOINT_PATH}/
mkdir -p ${TARGET_CHECKPOINT_PATH}/model_weights

cp "${SOURCE_CHECKPOINT_PATH}/common.pt" "${TARGET_CHECKPOINT_PATH}/model_weights/"
cp "${SOURCE_CHECKPOINT_PATH}/metadata.json" "${TARGET_CHECKPOINT_PATH}/model_weights/"
cp -r "${SOURCE_CHECKPOINT_PATH}/model"* "${TARGET_CHECKPOINT_PATH}/model_weights/"

echo "Checkpoint has been prepared at ${TARGET_CHECKPOINT_PATH}"