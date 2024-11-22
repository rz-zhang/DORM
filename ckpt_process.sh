#!/bin/bash

SOURCE_CHECKPOINT_PATH="/workspace/12b/NeMo-Aligner/results_model/1119_ec2_1node_dorm_8k_2epoch_lr3e6/checkpoints"
EXAMPLE_STRUCTURE_PATH="/workspace/NeMo-Aligner/results_model/rm_8b_inst_hs2/checkpoints/nemo_example"
TARGET_CHECKPOINT_PATH="/workspace/NeMo-Aligner/results_model/oh_mix_oa2_hs2_0717/checkpoints/nemo_ckpt_oh_mix_large_diff_step_300_consumed_samples_24000"

mkdir -p ${TARGET_CHECKPOINT_PATH}

cp ${EXAMPLE_STRUCTURE_PATH}/model_config.yaml ${TARGET_CHECKPOINT_PATH}/
mkdir -p ${TARGET_CHECKPOINT_PATH}/model_weights

cp "${SOURCE_CHECKPOINT_PATH}/common.pt" "${TARGET_CHECKPOINT_PATH}/model_weights/"
cp "${SOURCE_CHECKPOINT_PATH}/metadata.json" "${TARGET_CHECKPOINT_PATH}/model_weights/"
cp -r "${SOURCE_CHECKPOINT_PATH}/model"* "${TARGET_CHECKPOINT_PATH}/model_weights/"

echo "Checkpoint has been prepared at ${TARGET_CHECKPOINT_PATH}"