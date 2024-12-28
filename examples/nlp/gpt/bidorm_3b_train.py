import torch
import copy
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.bidorm import SupervisedTrainer, BilevelRewardModelTrainer, SampProb
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_regression_rm_datasets,
    build_train_valid_test_rm_datasets,
    build_train_valid_test_weighted_regression_rm_datasets,
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


from importlib import import_module

"""Script to start Reward Model training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="llama_3b")
def main(cfg) -> None:
    """Main training function modified to support weighted dataset training"""
    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Initialize trainer and exp manager
    trainer = resolve_and_create_trainer(cfg, "rm")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # Load model
    ptl_model = load_from_nemo(
        reward_model_cls,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )

    def print_trainable_params(model):
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
        print(f"Total trainable parameters: {total_params}")
    print_trainable_params(ptl_model)

    # Initialize second model for bilevel optimization if using weighted training
    use_weighted_training = cfg.model.get("use_weighted_training", False)
    if use_weighted_training:
        trainer_u = resolve_and_create_trainer(cfg, "rm")
        ptl_model_u = load_from_nemo(
            reward_model_cls,
            cfg.model,
            trainer_u,
            strict=True,
            load_base_model_only=True,
            restore_path=cfg.pretrained_checkpoint.restore_from_path,
        )

        # Initialize sampling probabilities
        # samp_prob = SampProb(num_partitions=len(WeightedRegressionRewardModelDataset.KNOWN_SOURCES))
        samp_prob = SampProb(num_partitions=5)
    else:
        ptl_model_u = None
        samp_prob = None

    # Get checkpoint info if exists
    trainer_restore_path = trainer.ckpt_path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    if use_weighted_training and ptl_model_u is not None:
        init_distributed(trainer_u, ptl_model_u, cfg.model.get("transformer_engine", False))

    # Use the entire dataset
    train_valid_test_num_samples = [-1 * cfg.model.global_batch_size] * 3

    # Choose appropriate dataset builder
    if reward_model_type == RewardModelType.REGRESSION:
        if use_weighted_training:
            dataset_builder = build_train_valid_test_weighted_regression_rm_datasets
        else:
            dataset_builder = build_train_valid_test_regression_rm_datasets
    else:
        raise ValueError(f"Only support regression reward model, but got {reward_model_type}")

    logging.info("\n\n************** Dataset Loading ***********")
    train_ds, validation_ds, test_ds = dataset_builder(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=ptl_model.tokenizer,
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

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    # Initialize optimizer for model_u and samp_prob if using weighted training
    def get_optimizer(model_params, optimizer_args_dict):
        """Gets optimizer given a configuration.

        Args:
            model_params: same type as model.parameters().
            optimizer_args_dict: a dict mapping optimizer arguments to their values.
                Except one key called "name", which specifies the optimizer name.
        """
        name = optimizer_args_dict['name']
        new_optimizer_args_dict = copy.deepcopy(optimizer_args_dict)
        new_optimizer_args_dict.pop('name', None)

        if name == 'sgd':
            return torch.optim.SGD(model_params, **new_optimizer_args_dict)
        elif name == 'adagrad':
            return torch.optim.Adagrad(model_params, **new_optimizer_args_dict)
        elif name == 'adam':
            return torch.optim.Adam(model_params, **new_optimizer_args_dict)
        elif name == 'adamw':
            return torch.optim.AdamW(model_params, **new_optimizer_args_dict)
        else:
            raise UnsupportedOptimizerError(f'Optimizer "{name}" is not supported')

    if use_weighted_training and ptl_model_u is not None:
        init_using_ptl(trainer_u, ptl_model_u, train_dataloader, train_ds)
        optimizer_u, _ = extract_optimizer_scheduler_from_ptl_model(ptl_model_u)
        optim_samp_config = {
            'name': 'adamw',  # or use cfg.model.get('optim_samp_name', 'adamw')
            'lr': cfg.model.get('optim_samp_lr', 0.001),
            'weight_decay': cfg.model.get('optim_samp_weight_decay', 0.0)
        }
        optimizer_samp = get_optimizer(samp_prob.parameters(), optim_samp_config)
    else:
        optimizer_u = None
        optimizer_samp = None

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    # Choose appropriate trainer class
    if use_weighted_training:
        rm_trainer_cls = BilevelRewardModelTrainer
        trainer_kwargs = {
            'model_u': ptl_model_u,
            'optimizer_u': optimizer_u,
            'samp_prob': samp_prob,
            'optimizer_samp': optimizer_samp,
            'tau': cfg.model.get('tau', 1.0),
            'init_alpha': cfg.model.get('init_alpha', 1.0),
        }
    else:
        rm_trainer_cls = SupervisedTrainer
        trainer_kwargs = {}

    rm_trainer = rm_trainer_cls(
        cfg=cfg.trainer.rm,
        model_w=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
        **trainer_kwargs
    )

    if custom_trainer_state_dict is not None:
        rm_trainer.load_state_dict(custom_trainer_state_dict)

    logging.info("\n\n************** Training Start! ***********")
    rm_trainer.fit()

if __name__ == "__main__":
    main()