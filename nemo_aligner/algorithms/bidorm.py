import torch
import torch.nn as nn
from omegaconf.omegaconf import OmegaConf
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.train_utils import clip_gradients

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
        self.model_u = model_u
        self.samp_prob = samp_prob
        self.optimizer_u = optimizer_u
        self.optimizer_samp = optimizer_samp
        self.tau = tau
        self.alpha = init_alpha
        self.current_batch = None
        # Buffer for collecting samples per partition
        self.micro_batch_size = model_w.cfg.micro_batch_size
        self.partition_buffers = {i: [] for i in range(len(samp_prob.prob))}
        self.buffer_size = self.micro_batch_size * 4  # Collect enough samples

    def train_single_step(self, batch):
        """Override train_single_step to implement bilevel optimization"""
        self.alpha *= self.tau
        self.current_batch = batch

        # Step 1: Update model_u to minimize -alpha * L2
        self.optimizer_u.zero_grad()
        self.model_u.prepare_for_training_step()
        loss_u_L2, metrics_u = self.model_u.get_loss_and_metrics(
            batch=self.current_batch,
            forward_only=False
        )
        # Convert float loss to tensor if needed and scale it
        if isinstance(loss_u_L2, float):
            loss_u_L2 = torch.tensor(loss_u_L2, device=self.model_u.device, requires_grad=True)
        scaled_loss_u = -self.alpha * loss_u_L2  # Negative for minimization
        scaled_loss_u.backward()
        self.optimizer_u.step()
        self.model_u.finish_training_step()

        # Step 2: Update model_w to minimize L1 + alpha * L2
        self.optimizer.zero_grad()
        self.model.prepare_for_training_step()

        # Training loss (L2)
        loss_w_L2, metrics_w = self.model.get_loss_and_metrics(
            batch=self.current_batch,
            forward_only=False
        )
        if isinstance(loss_w_L2, float):
            loss_w_L2 = torch.tensor(loss_w_L2, device=self.model.device, requires_grad=True)

        # Validation loss (L1)
        try:
            val_batch = next(iter(self.val_dataloader))
        except StopIteration:
            val_dataloader_iter = iter(self.val_dataloader)
            val_batch = next(val_dataloader_iter)

        loss_w_L1, metrics_val = self.model.get_loss_and_metrics(
            batch=val_batch,
            forward_only=False
        )
        if isinstance(loss_w_L1, float):
            loss_w_L1 = torch.tensor(loss_w_L1, device=self.model.device, requires_grad=True)

        # Scale L2 loss and combine with L1
        total_loss = loss_w_L1 + self.alpha * loss_w_L2
        total_loss.backward()

        # Get gradient norm before optimizer step
        grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        self.optimizer.step()
        self.scheduler.step()
        self.model.finish_training_step()

        # Step 3: Update sampling probabilities
        self.optimizer_samp.zero_grad()
        grad_p = self._compute_prob_gradients(self.current_batch)
        self.samp_prob.p.grad = grad_p
        self.optimizer_samp.step()

        # Get learning rate for metrics
        lr = self.optimizer.param_groups[0]["lr"]

        # Combine all metrics
        trainer_metrics = {
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "lr": lr,
            "loss": total_loss.item(),
            "loss_L1": loss_w_L1.item(),
            "loss_L2_w": loss_w_L2.item(),
            "loss_L2_u": loss_u_L2.item(),
            "alpha": self.alpha,
        }

        # Log data weights
        for source, partition_id in self.train_dataloader.dataset.KNOWN_SOURCES.items():
            weight = self.samp_prob.prob[partition_id].item()
            trainer_metrics[f"data_weight/{source}"] = weight

        return total_loss.item(), trainer_metrics

    def _compute_prob_gradients(self, batch):
        """Compute gradients for sampling probabilities"""
        samp_prob = self.samp_prob.prob
        s = samp_prob.reshape(-1, 1)
        jacobian = torch.diagflat(s) - torch.mul(s, s.T)

        with torch.no_grad():
            L2_u_grad = self._get_prob_loss_grad(self.model_u, batch)
            L2_w_grad = self._get_prob_loss_grad(self.model, batch)
            grad_p = L2_w_grad + L2_u_grad

        return torch.matmul(grad_p, jacobian)

    # def _get_prob_loss_grad(self, model, batch):
    #     # If micro batch size is 1, we can compute gradients for each partition separately
    #     """Helper to compute loss gradients w.r.t probabilities for a model"""
    #     losses = []
    #     for partition_idx in range(len(self.samp_prob.prob)):
    #         partition_mask = (batch['partition_id'] == partition_idx)
    #         if partition_mask.any():
    #             partition_batch = {k: v[partition_mask] for k, v in batch.items()}
    #             loss, _ = model.get_loss_and_metrics(partition_batch, forward_only=True)
    #             if isinstance(loss, float):
    #                 loss = torch.tensor(loss, device=model.device)
    #             losses.append(loss)
    #         else:
    #             losses.append(torch.tensor(0.0, device=model.device))
    #     return torch.stack(losses)

    # def _get_prob_loss_grad(self, model, batch):
    #     # Add samples to buffers to handle partitions that not being divisible by micro_batch_size
    #     for partition_idx in range(len(self.samp_prob.prob)):
    #         partition_mask = (batch['partition_id'] == partition_idx)
    #         if partition_mask.any():
    #             partition_data = {k: v[partition_mask] for k, v in batch.items()}
    #             self.partition_buffers[partition_idx].append(partition_data)
    #
    #     # Only compute gradients when buffers are full
    #     losses = []
    #     for partition_idx, buffer in self.partition_buffers.items():
    #         if len(buffer) * self.micro_batch_size >= self.buffer_size:
    #             # Combine collected samples
    #             combined_batch = {
    #                 k: torch.cat([b[k] for b in buffer], dim=0)
    #                 for k in buffer[0].keys()
    #             }
    #             # Compute loss on full batch
    #             loss, _ = model.get_loss_and_metrics(combined_batch, forward_only=True)
    #             losses.append(loss)
    #             # Clear buffer
    #             buffer.clear()
    #         else:
    #             losses.append(None)
    #
    #     # Return zeros for partitions that don't have enough samples yet
    #     device = batch['partition_id'].device
    #     return torch.stack([loss if loss is not None else torch.tensor(0.0, device=device)
    #                       for loss in losses])

    # def _get_prob_loss_grad(self, model, batch):
    #     """Helper to compute loss gradients w.r.t probabilities for a model"""
    #     losses = []
    #     device = batch['partition_id'].device
    #
    #     for partition_idx in range(len(self.samp_prob.prob)):
    #         partition_mask = (batch['partition_id'] == partition_idx)
    #         if partition_mask.any():
    #             # Get number of samples in this partition
    #             num_samples = partition_mask.sum().item()
    #
    #             # Get samples that can form complete micro-batches
    #             num_valid_samples = (num_samples // self.micro_batch_size) * self.micro_batch_size
    #
    #             if num_valid_samples > 0:
    #                 # Only take samples that form complete micro-batches
    #                 valid_indices = torch.where(partition_mask)[0][:num_valid_samples]
    #                 partition_batch = {k: v[valid_indices] for k, v in batch.items()}
    #
    #                 loss, _ = model.get_loss_and_metrics(partition_batch, forward_only=True)
    #                 losses.append(loss)
    #             else:
    #                 # Not enough samples for a complete micro-batch
    #                 losses.append(torch.tensor(0.0, device=device))
    #         else:
    #             losses.append(torch.tensor(0.0, device=device))
    #
    #     return torch.stack(losses)
    # At the start of the function
    # from megatron.core.num_microbatches_calculator import get_num_microbatches


    # def _get_prob_loss_grad(self, model, batch):
    #     """Helper to compute loss gradients w.r.t probabilities for a model"""
    #     losses = []
    #     device = batch['partition_id'].device
    #
    #     # Debug total batch info
    #     first_key = next(iter(batch.keys()))
    #     print(f"Total batch size before partition: {batch[first_key].shape[0]}")
    #     print(f"Using micro_batch_size: {self.micro_batch_size}")
    #     print(f"Unique partition_ids in batch: {torch.unique(batch['partition_id'])}")
    #
    #     for partition_idx in range(len(self.samp_prob.prob)):
    #         partition_mask = (batch['partition_id'] == partition_idx)
    #
    #         # Debug partition info
    #         num_samples = partition_mask.sum().item()
    #         print(f"\nProcessing partition {partition_idx}")
    #         print(f"Number of samples in partition: {num_samples}")
    #
    #         if partition_mask.any():
    #             num_valid_samples = (num_samples // self.micro_batch_size) * self.micro_batch_size
    #             print(f"Number of valid samples (divisible by micro_batch_size): {num_valid_samples}")
    #
    #             if num_valid_samples > 0:
    #                 valid_indices = torch.where(partition_mask)[0][:num_valid_samples]
    #                 partition_batch = {k: v[valid_indices] for k, v in batch.items()}
    #
    #                 # Debug partition batch
    #                 print(f"Partition batch sizes:")
    #                 for k, v in partition_batch.items():
    #                     print(f"  {k}: {v.shape}")
    #
    #                 try:
    #                     #print(f"Megatron get_num_microbatches() returns: {get_num_microbatches()}")
    #                     loss, _ = model.get_loss_and_metrics(partition_batch, forward_only=True)
    #                     print(f"Successfully computed loss for partition {partition_idx}")
    #                     losses.append(loss)
    #                 except Exception as e:
    #                     print(f"Error computing loss for partition {partition_idx}: {str(e)}")
    #                     print(f"Error type: {type(e)}")
    #                     losses.append(torch.tensor(0.0, device=device))
    #             else:
    #                 print(f"No valid samples for partition {partition_idx}")
    #                 losses.append(torch.tensor(0.0, device=device))
    #         else:
    #             print(f"No samples found for partition {partition_idx}")
    #             losses.append(torch.tensor(0.0, device=device))
    #
    #     print("\nFinal number of losses:", len(losses))
    #     return torch.stack(losses)

    def _get_prob_loss_grad(self, model, batch):
        """Helper to compute loss gradients w.r.t probabilities for a model"""
        losses = []
        device = batch['partition_id'].device

        # Debug total batch info
        first_key = next(iter(batch.keys()))
        print(f"Total batch size before partition: {batch[first_key].shape[0]}")
        print(f"Using micro_batch_size: {self.micro_batch_size}")
        print(f"Unique partition_ids in batch: {torch.unique(batch['partition_id'])}")

        for partition_idx in range(len(self.samp_prob.prob)):
            partition_mask = (batch['partition_id'] == partition_idx)

            # Debug partition info
            num_samples = partition_mask.sum().item()
            print(f"\nProcessing partition {partition_idx}")
            print(f"Number of samples in partition: {num_samples}")

            if partition_mask.any():
                num_valid_samples = (num_samples // self.micro_batch_size) * self.micro_batch_size
                print(f"Number of valid samples (divisible by micro_batch_size): {num_valid_samples}")

                if num_valid_samples > 0:
                    valid_indices = torch.where(partition_mask)[0][:num_valid_samples]
                    partition_batch = {k: v[valid_indices] for k, v in batch.items()}

                    # Debug partition batch
                    print(f"Partition batch sizes:")
                    for k, v in partition_batch.items():
                        print(f"  {k}: {v.shape}")

                    try:
                        loss, _ = model.get_loss_and_metrics(partition_batch, forward_only=True)
                        # Convert loss to tensor if it's a float
                        if isinstance(loss, float):
                            loss = torch.tensor(loss, device=device)
                        print(f"Successfully computed loss for partition {partition_idx}")
                        losses.append(loss)
                    except Exception as e:
                        print(f"Error computing loss for partition {partition_idx}: {str(e)}")
                        print(f"Error type: {type(e)}")
                        losses.append(torch.tensor(0.0, device=device))
                else:
                    print(f"No valid samples for partition {partition_idx}")
                    losses.append(torch.tensor(0.0, device=device))
            else:
                print(f"No samples found for partition {partition_idx}")
                losses.append(torch.tensor(0.0, device=device))

        print("\nFinal number of losses:", len(losses))
        # Make sure all losses are tensors before stacking
        losses = [loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=device) for loss in losses]
        return torch.stack(losses)
