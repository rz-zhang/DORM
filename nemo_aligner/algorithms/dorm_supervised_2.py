import torch
import torch.nn as nn
from omegaconf.omegaconf import OmegaConf
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType

import wandb


class SampProb(nn.Module):
    """Sampling probability module from minmax.py"""

    def __init__(self, num_partitions):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(num_partitions))

    @property
    def prob(self):
        return torch.softmax(self.p, dim=0)


class BilevelRewardModelTrainer(SupervisedTrainer):
    """Extended trainer implementing bilevel optimization for reward model training"""

    def __init__(self, cfg, model_w, model_u, samp_prob, optimizer_u, optimizer_samp, tau, init_alpha, *args, **kwargs):
        super().__init__(cfg, model_w, *args, **kwargs)
        self.model_u = model_u  # Second model for bilevel optimization
        self.samp_prob = samp_prob  # Sampling probabilities
        self.optimizer_u = optimizer_u
        self.optimizer_samp = optimizer_samp
        self.tau = tau
        self.alpha = init_alpha
        self.global_step = 0

    def training_step(self, batch):
        """Override training step to implement bilevel optimization"""
        self.alpha *= self.tau

        # Step 1: Update model_u to minimize -alpha * L2
        self.optimizer_u.zero_grad()
        loss_u_L2, metrics_u = self.model_u.get_loss_and_metrics(
            batch=batch,
            forward_only=False,
            loss_scale=-self.alpha
        )
        loss_u_L2.backward()
        self.optimizer_u.step()

        # Step 2: Update model_w to minimize L1 + alpha * L2
        self.optimizer.zero_grad()
        # Training loss (L2)
        loss_w_L2, metrics_w = self.model.get_loss_and_metrics(
            batch=batch,
            forward_only=False,
            loss_scale=self.alpha
        )
        # Validation loss (L1)
        val_batch = next(iter(self.val_dataloader))
        loss_w_L1, metrics_val = self.model.get_loss_and_metrics(
            batch=val_batch,
            forward_only=False,
            loss_scale=1.0
        )
        total_loss = loss_w_L1 + loss_w_L2
        total_loss.backward()
        self.optimizer.step()

        # Step 3: Update sampling probabilities
        self.optim_samp.zero_grad()
        grad_p = self._compute_prob_gradients(batch)
        self.samp_prob.p.grad = grad_p
        self.optim_samp.step()

        # Combine metrics
        metrics = {
            "loss_L1": loss_w_L1.item(),
            "loss_L2_w": loss_w_L2.item(),
            "loss_L2_u": loss_u_L2.item(),
            **metrics_w,
            **metrics_val
        }

        # Log data weights to wandb
        for source, partition_id in self.train_dataloader.dataset.KNOWN_SOURCES.items():
            weight = self.samp_prob.prob[partition_id].item()
            # Use source name in metric name for clarity
            self.logger.log_metrics({f"data_weights/{source}": weight}, step=self.global_step)

        # You can also log alpha
        self.logger.log_metrics({"training/alpha": self.alpha}, step=self.global_step)

        # Log as a single chart showing all weights
        weight_dict = {
            source: self.samp_prob.prob[partition_id].item()
            for source, partition_id in self.train_dataloader.dataset.KNOWN_SOURCES.items()
        }
        self.logger.log_metrics({"data_weights": weight_dict}, step=self.global_step)
        self.global_step += 1

        return total_loss, metrics

    def _compute_prob_gradients(self, batch):
        """Compute gradients for sampling probabilities"""
        # Get probability distribution
        samp_prob = self.samp_prob.prob

        # Compute Jacobian
        s = samp_prob.reshape(-1, 1)
        jacobian = torch.diagflat(s) - torch.mul(s, s.T)

        # Compute gradients from losses
        with torch.no_grad():
            L2_u_grad = self._get_prob_loss_grad(self.model_u, batch)
            L2_w_grad = self._get_prob_loss_grad(self.model, batch)
            grad_p = L2_w_grad + L2_u_grad

        return torch.matmul(grad_p, jacobian)

    def _get_prob_loss_grad(self, model, batch):
        """Helper to compute loss gradients w.r.t probabilities for a model"""
        losses = []
        for partition_idx in range(len(self.samp_prob.prob)):
            # Compute loss for each partition
            partition_mask = (batch['partition_ids'] == partition_idx)
            if partition_mask.any():
                partition_batch = {k: v[partition_mask] for k, v in batch.items()}
                loss, _ = model.get_loss_and_metrics(partition_batch, forward_only=True)
                losses.append(loss)
            else:
                losses.append(torch.tensor(0.0, device=self.device))
        return torch.stack(losses)


def main(cfg):
    """Main training function with bilevel optimization"""
    # Standard NeMo setup
    trainer = resolve_and_create_trainer(cfg, "rm")
    exp_manager(trainer, cfg.exp_manager)

    # Create models
    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    model_w = load_from_nemo(reward_model_cls, cfg.model, trainer)
    model_u = copy.deepcopy(model_w)  # Create second model for bilevel opt

    # Create sampling probability module
    num_partitions = cfg.data.num_partitions  # Add to config
    samp_prob = SampProb(num_partitions)

    # Create optimizers
    optim_w = get_optimizer(model_w.parameters(), cfg.optim)
    optim_u = get_optimizer(model_u.parameters(), cfg.optim)
    optim_samp = get_optimizer(samp_prob.parameters(), cfg.optim_samp)

    # Create bilevel trainer
    bilevel_trainer = BilevelRewardModelTrainer(
        cfg=cfg,
        model_w=model_w,
        model_u=model_u,
        samp_prob=samp_prob,
        optimizer=optim_w,
        optimizer_u=optim_u,
        optim_samp=optim_samp,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        logger=logger
    )

    # Train
    bilevel_trainer.fit()


if __name__ == "__main__":
    @hydra_runner(config_path="conf", config_name="llama_rm_train_bilevel")
    def main_wrapper(cfg):
        main(cfg)


    main_wrapper()