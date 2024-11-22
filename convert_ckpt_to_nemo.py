import torch
from omegaconf import OmegaConf
from nemo.utils import logging
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.utils import load_from_nemo

def convert_checkpoint_to_nemo(
    checkpoint_dir: str,
    output_nemo_file: str,
    tensor_model_parallel_size: int = 8,
    pipeline_model_parallel_size: int = 1,
):
    logging.info(f"Converting checkpoint at {checkpoint_dir} to .nemo format")

    # 加载配置
    config_file = os.path.join(checkpoint_dir, "model_config.yaml")
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found at {config_file}")

    cfg = OmegaConf.load(config_file)

    # 设置并行参数
    cfg.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.pipeline_model_parallel_size = pipeline_model_parallel_size

    # 获取正确的模型类
    reward_model_type = RewardModelType(cfg.get("reward_model_type", "regression"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    try:
        # 使用与训练脚本相同的加载方式
        model = load_from_nemo(
            reward_model_cls,
            cfg,
            trainer=None,  # 转换时不需要trainer
            strict=True,
            load_base_model_only=False,  # 加载完整模型
            restore_path=checkpoint_dir,
        )

        # 保存为.nemo格式
        model.save_to(output_nemo_file)
        logging.info(f"Successfully converted and saved model to {output_nemo_file}")

    except Exception as e:
        logging.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置路径
    checkpoint_dir = "/workspace/NeMo-Aligner/results_model/1119_ec2_1node_dorm_8k_2epoch_lr3e6/checkpoints/nemo_ckpt_consumed_samples_128000"
    output_nemo_file = "/workspace/NeMo-Aligner/results_model/1119_ec2_1node_dorm_8k_2epoch_lr3e6/checkpoints/model_128000.nemo"

    # 使用分布式环境下的转换
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            convert_checkpoint_to_nemo(
                checkpoint_dir=checkpoint_dir,
                output_nemo_file=output_nemo_file,
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=1
            )
        torch.distributed.barrier()
    else:
        convert_checkpoint_to_nemo(
            checkpoint_dir=checkpoint_dir,
            output_nemo_file=output_nemo_file,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1
        )