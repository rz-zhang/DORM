from typing import Dict, Optional, Any
import torch
from torch.nn import Parameter
from omegaconf.dictconfig import DictConfig
from collections import defaultdict
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.utils.train_utils import clip_gradients

class ScaleBiOTrainer(SupervisedTrainer):
    """Bilevel optimization trainer that extends NeMo's SupervisedTrainer.

    This trainer implements the ScaleBiO algorithm for data reweighting using
    bilevel optimization within NeMo-Aligner's training framework.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_w,  # Main model
        model_u,  # Auxiliary model for minimax
        optimizer_w,  # Main model optimizer
        optimizer_u,  # Auxiliary model optimizer
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
        run_init_validation: bool = False,
    ):
        # Initialize parent class
        super().__init__(
            cfg=cfg,
            model=model_w,  # Parent class will manage main model
            optimizer=optimizer_w,
            scheduler=None,  # We'll manage schedulers separately
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            logger=logger,
            ckpt_callback=ckpt_callback,
            run_timer=run_timer,
            run_init_validation=run_init_validation,
        )

        # ScaleBiO specific components
        self.model_u = model_u
        self.optimizer_u = optimizer_u

        # Initialize sampling probabilities
        num_sources = cfg.get("num_data_sources", len(train_dataloader.dataset.source_map))
        self.samp_prob = Parameter(torch.zeros(num_sources, device=model_w.device))
        self.optimizer_samp = torch.optim.AdamW([self.samp_prob], lr=cfg.samp_prob_lr)

        # ScaleBiO hyperparameters
        self.alpha = cfg.get("alpha", 100.0)  # Penalty parameter
        self.tau = cfg.get("tau", 0.9)  # Decay factor

        # Additional metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.num_outer_iter = cfg.get("num_outer_iter", 3)  # N in Algorithm 1
        self.num_inner_iter = cfg.get("num_inner_iter", None)  # K in Algorithm 1

        if self.num_inner_iter is None:
            # If not specified, set to dataset size / batch size * epochs
            self.num_inner_iter = len(train_dataloader.dataset) // cfg.global_batch_size * cfg.max_epochs

    def train_single_step(self, batch: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Override train step to implement bilevel optimization."""
        # Initialize metrics
        metrics = {}

        # Step 1: Update auxiliary model u (maximize)
        self.optimizer_u.zero_grad()
        self.model_u.train()

        # Get loss for model u with negative scaling (for maximization)
        loss_u, metrics_u = self.model_u.get_loss_and_metrics(
            batch=batch,
            forward_only=False,
            loss_scale=-self.alpha/self.cfg.gradient_accumulation_steps
        )

        # Apply sampling weights if source information available
        if "source_ids" in batch:
            loss_u = loss_u * torch.gather(
                self.samp_prob,
                0,
                batch["source_ids"].to(self.samp_prob.device)
            )
            loss_u = loss_u.mean()

        grad_norm_u = clip_gradients(self.model_u, self.cfg.gradient_clip_val)
        self.optimizer_u.step()

        # Step 2: Update main model w (minimize)
        self.optimizer.zero_grad()
        self.model.train()

        # Get training loss
        loss_w_train, metrics_w = self.model.get_loss_and_metrics(
            batch=batch,
            forward_only=False,
            loss_scale=self.alpha/self.cfg.gradient_accumulation_steps
        )

        # Apply sampling weights
        if "source_ids" in batch:
            loss_w_train = loss_w_train * torch.gather(
                self.samp_prob,
                0,
                batch["source_ids"].to(self.samp_prob.device)
            )
            loss_w_train = loss_w_train.mean()

        # Get validation loss
        val_batch = next(iter(self.val_dataloader))
        loss_w_val, metrics_v = self.model.get_loss_and_metrics(
            batch=val_batch,
            forward_only=False,
            loss_scale=1.0/self.cfg.gradient_accumulation_steps
        )

        # Combined loss for model w
        loss_w = loss_w_train + loss_w_val

        grad_norm_w = clip_gradients(self.model, self.cfg.gradient_clip_val)
        self.optimizer.step()

        # Step 3: Update sampling probabilities
        self.optimizer_samp.zero_grad()

        if "source_ids" in batch:
            # Compute gradients for sampling weights
            grad = torch.autograd.grad(
                outputs=loss_w,
                inputs=self.samp_prob,
                retain_graph=True
            )[0]

            # Update with softmax to ensure valid probability distribution
            with torch.no_grad():
                self.samp_prob.data = torch.softmax(
                    self.samp_prob - self.optimizer_samp.param_groups[0]["lr"] * grad,
                    dim=0
                )

        # Collect metrics
        metrics.update({
            "loss": (loss_w + loss_u).item(),
            "loss_w_train": loss_w_train.item(),
            "loss_w_val": loss_w_val.item(),
            "loss_u": loss_u.item(),
            "grad_norm_w": grad_norm_w,
            "grad_norm_u": grad_norm_u,
            "lr": self.optimizer.param_groups[0]["lr"],
            "alpha": self.alpha,
            **metrics_u,
            **metrics_w,
            **metrics_v,
        })

        # Update alpha
        self.alpha *= self.tau

        return loss_w.item() + loss_u.item(), metrics

    def validation_step(self, batch: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Override validation step to evaluate both models."""
        self.model.eval()
        self.model_u.eval()

        with torch.no_grad():
            # Validate main model
            loss_w, metrics_w = self.model.get_loss_and_metrics(
                batch=batch,
                forward_only=True
            )

            # Validate auxiliary model
            loss_u, metrics_u = self.model_u.get_loss_and_metrics(
                batch=batch,
                forward_only=True
            )

            metrics = {
                "loss_w": loss_w.item(),
                "loss_u": loss_u.item(),
                **metrics_w,
                **metrics_u,
            }

        return loss_w.item() + loss_u.item(), metrics

    def state_dict(self) -> Dict[str, Any]:
        """Save additional ScaleBiO state."""
        state = super().state_dict()
        state.update({
            "model_u_state": self.model_u.state_dict(),
            "optimizer_u_state": self.optimizer_u.state_dict(),
            "samp_prob": self.samp_prob.data,
            "optimizer_samp_state": self.optimizer_samp.state_dict(),
            "alpha": self.alpha,
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load additional ScaleBiO state."""
        super().load_state_dict(state_dict)
        self.model_u.load_state_dict(state_dict["model_u_state"])
        self.optimizer_u.load_state_dict(state_dict["optimizer_u_state"])
        self.samp_prob.data = state_dict["samp_prob"]
        self.optimizer_samp.load_state_dict(state_dict["optimizer_samp_state"])
        self.alpha = state_dict["alpha"]

    def fit(self):
        """
        Implements ScaleBiO's bilevel optimization training loop (Algorithm 1).
        Consists of outer iterations (N) each containing inner iterations (K).
        """
        if self.run_init_validation:
            self.logger.info("Running initial validation.")
            val_loss, val_metrics = self.run_validation()
            self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")

        self.run_timer.start_time()

        # Outer loop (N iterations)
        for i in range(self.num_outer_iter):
            self.logger.info(f"Starting outer iteration {i+1}/{self.num_outer_iter}")

            # Update alpha with decay
            self.alpha *= self.tau

            # Inner loop (K iterations)
            train_pbar = tqdm(
                range(self.num_inner_iter),
                desc=f"Outer iter {i+1}, Inner steps",
                leave=True
            )

            for k in train_pbar:
                # Check if we should stop
                if self.run_timer.is_finished():
                    self.logger.info("Time limit reached - stopping training")
                    return

                try:
                    batch = next(self.train_dataloader_iter)
                except (StopIteration, AttributeError):
                    self.train_dataloader_iter = iter(self.train_dataloader)
                    batch = next(self.train_dataloader_iter)

                # Update learning rates
                lr = (self.tau**(-i)) * self._get_lr_schedule(
                    self.step,
                    self.num_outer_iter * self.num_inner_iter
                )

                lr_u = lr * self.cfg.get("lr_u_factor", 1.0)
                lr_w = lr * self.cfg.get("lr_w_factor", 1.0)
                lr_lambda = lr * self.cfg.get("lr_lambda_factor", 1.0)

                self._update_lr(self.optimizer_u, lr_u)
                self._update_lr(self.optimizer, lr_w)
                self._update_lr(self.optimizer_samp, lr_lambda)

                # Execute bilevel optimization step
                self.timer.start("train_step_time")
                loss, metrics = self.train_single_step(batch)
                self.timer.stop("train_step_time")
                train_step_time = self.timer.get("train_step_time")

                # Update tracking
                self.consumed_samples += self.cfg.global_batch_size
                metrics["consumed_samples"] = self.consumed_samples
                metrics["step_time"] = train_step_time
                metrics["outer_iter"] = i + 1
                metrics["inner_iter"] = k + 1

                # Log metrics
                self.logger.log_metrics(metrics, step=self.step, prefix="train/")

                # Update progress bar
                train_pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "alpha": f"{self.alpha:.2f}"
                })

                # Run validation if needed
                if k % self.val_check_interval == 0:
                    val_loss, val_metrics = self.run_validation()
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")

                # Save checkpoint if needed
                if k % self.cfg.save_interval == 0:
                    self.save(metrics)

                self.step += 1

        # Final validation
        val_loss, val_metrics = self.run_validation()
        self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")

        # Final save
        self.save(metrics, is_train_end=True)
        self.logger.info("Training completed")

    def _get_lr_schedule(self, current_step: int, total_steps: int) -> float:
        """Simple linear learning rate schedule."""
        return self.cfg.init_lr * (1 - current_step / total_steps)

    def _update_lr(self, optimizer: torch.optim.Optimizer, new_lr: float):
        """Update learning rate for an optimizer."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr