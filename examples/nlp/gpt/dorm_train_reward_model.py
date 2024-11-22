import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_regression_rm_datasets,
    build_train_valid_test_rm_datasets,
)
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo

@hydra_runner(config_path="conf", config_name="training_rm")
def main(cfg) -> None:
    """
    Modified to support bilevel optimization for reward model training.
    """
    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "rm")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # Load both main and auxiliary models
    model_w = load_from_nemo(
        reward_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    # Create auxiliary model with same architecture
    model_u = load_from_nemo(
        reward_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    # Get trainer restore path and consumed samples
    trainer_restore_path = trainer.ckpt_path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, model_w, cfg.model.get("transformer_engine", False))
    init_distributed(trainer, model_u, cfg.model.get("transformer_engine", False))

    # Build datasets and dataloaders
    train_valid_test_num_samples = [-1 * cfg.model.global_batch_size] * 3

    if reward_model_type == RewardModelType.BINARY_RANKING:
        dataset_builder = build_train_valid_test_rm_datasets
    elif reward_model_type == RewardModelType.REGRESSION:
        dataset_builder = build_train_valid_test_regression_rm_datasets
    else:
        raise ValueError(f"Only support binary_ranking and regression reward model, but get {reward_model_type}")

    train_ds, validation_ds, test_ds = dataset_builder(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=model_w.tokenizer,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
        use_random_sampler=False,
    )

    # Initialize both models with dataloaders
    init_using_ptl(trainer, model_w, train_dataloader, train_ds)
    init_using_ptl(trainer, model_u, train_dataloader, train_ds)

    # Get optimizers for both models
    optimizer_w, scheduler_w = extract_optimizer_scheduler_from_ptl_model(model_w)
    optimizer_u, scheduler_u = extract_optimizer_scheduler_from_ptl_model(model_u)

    ckpt_callback = add_custom_checkpoint_callback(trainer, model_w)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    # Use ScaleBiO trainer instead of SupervisedTrainer
    rm_trainer = ScaleBiOTrainer(
        cfg=cfg.trainer.rm,
        model_w=model_w,
        model_u=model_u,
        optimizer_w=optimizer_w,
        optimizer_u=optimizer_u,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        rm_trainer.load_state_dict(custom_trainer_state_dict)

    rm_trainer.fit()

if __name__ == "__main__":
    main()
