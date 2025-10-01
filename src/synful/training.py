"""
Modern training module for Synful PyTorch.

Provides PyTorch Lightning trainers with modern features:
- Mixed precision training
- Automatic batch size finding
- Learning rate scheduling
- Advanced metrics and logging
- Multi-GPU support
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    BatchSizeFinder, ModelSummary
)
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,
    BinaryJaccardIndex
)
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from .models import UNet3D
from .synapse import SynapseCollection
from .data import SynfulAugmentations, create_training_transforms
from .snapshot import SnapshotCallback, SnapshotManager

logger = logging.getLogger(__name__)


class SynfulLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for Synful training.
    
    Supports both single-task (mask only) and multi-task (mask + direction)
    training with modern optimization and metrics.
    """
    
    def __init__(
        self,
        model: UNet3D,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # "cosine", "plateau", "step"
        mask_loss_weight: float = 1.0,
        direction_loss_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        augment_prob: float = 0.8,
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: UNet3D model instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler: Learning rate scheduler type
            mask_loss_weight: Weight for mask prediction loss
            direction_loss_weight: Weight for direction prediction loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            augment_prob: Probability of applying augmentations
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.mask_loss_weight = mask_loss_weight
        self.direction_loss_weight = direction_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Setup augmentations
        self.augmentations = SynfulAugmentations(
            prob_geometric=augment_prob * 0.8,
            prob_intensity=augment_prob * 0.7,
            prob_noise=augment_prob * 0.5,
        )
        
        # Setup metrics
        self._setup_metrics()
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def _setup_metrics(self):
        """Setup training and validation metrics."""
        mask_metrics = MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1': BinaryF1Score(),
            'iou': BinaryJaccardIndex(),
        })
        
        direction_metrics = MetricCollection({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
        })
        
        self.train_mask_metrics = mask_metrics.clone(prefix='train/mask_')
        self.val_mask_metrics = mask_metrics.clone(prefix='val/mask_')
        
        if self.model.multitask:
            self.train_dir_metrics = direction_metrics.clone(prefix='train/dir_')
            self.val_dir_metrics = direction_metrics.clone(prefix='val/dir_')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        raw = batch['raw']
        mask_target = batch['mask']
        
        # Apply augmentations
        if self.training:
            aug_data = {'raw': raw, 'mask': mask_target}
            if 'direction' in batch:
                aug_data['direction'] = batch['direction']
            
            aug_data = self.augmentations(
                aug_data['raw'], 
                aug_data['mask'],
                aug_data.get('direction')
            )
            raw = aug_data['raw']
            mask_target = aug_data['mask']
            if 'direction' in aug_data:
                direction_target = aug_data['direction']
        else:
            direction_target = batch.get('direction')
        
        # Forward pass
        outputs = self(raw)
        mask_pred = outputs['mask_logits']
        
        # Mask loss
        mask_loss = self.focal_loss(mask_pred, mask_target)
        total_loss = self.mask_loss_weight * mask_loss
        
        # Direction loss (if multitask)
        if self.model.multitask and direction_target is not None:
            direction_pred = outputs['direction_vectors']
            direction_loss = self.mse_loss(direction_pred, direction_target)
            total_loss += self.direction_loss_weight * direction_loss
            
            # Log direction loss
            self.log('train/direction_loss', direction_loss, prog_bar=False)
            
            # Update direction metrics
            direction_pred_flat = direction_pred.view(-1, direction_pred.size(1))
            direction_target_flat = direction_target.view(-1, direction_target.size(1))
            self.train_dir_metrics.update(direction_pred_flat, direction_target_flat)
        
        # Log mask loss
        self.log('train/mask_loss', mask_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        # Update mask metrics
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        self.train_mask_metrics.update(mask_pred_sigmoid, mask_target.int())
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        raw = batch['raw']
        mask_target = batch['mask']
        direction_target = batch.get('direction')
        
        # Forward pass
        outputs = self(raw)
        mask_pred = outputs['mask_logits']
        
        # Mask loss
        mask_loss = self.focal_loss(mask_pred, mask_target)
        total_loss = self.mask_loss_weight * mask_loss
        
        # Direction loss (if multitask)
        if self.model.multitask and direction_target is not None:
            direction_pred = outputs['direction_vectors']
            direction_loss = self.mse_loss(direction_pred, direction_target)
            total_loss += self.direction_loss_weight * direction_loss
            
            # Log direction loss
            self.log('val/direction_loss', direction_loss, prog_bar=False)
            
            # Update direction metrics
            direction_pred_flat = direction_pred.view(-1, direction_pred.size(1))
            direction_target_flat = direction_target.view(-1, direction_target.size(1))
            self.val_dir_metrics.update(direction_pred_flat, direction_target_flat)
        
        # Log losses
        self.log('val/mask_loss', mask_loss, prog_bar=True)
        self.log('val/total_loss', total_loss, prog_bar=True)
        
        # Update mask metrics
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        self.val_mask_metrics.update(mask_pred_sigmoid, mask_target.int())
        
        return total_loss
    
    def on_training_epoch_end(self):
        """Log training metrics at epoch end."""
        # Compute and log mask metrics
        mask_metrics = self.train_mask_metrics.compute()
        self.log_dict(mask_metrics, prog_bar=False)
        self.train_mask_metrics.reset()
        
        # Compute and log direction metrics (if multitask)
        if self.model.multitask:
            dir_metrics = self.train_dir_metrics.compute()
            self.log_dict(dir_metrics, prog_bar=False)
            self.train_dir_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        # Compute and log mask metrics
        mask_metrics = self.val_mask_metrics.compute()
        self.log_dict(mask_metrics, prog_bar=True)
        self.val_mask_metrics.reset()
        
        # Compute and log direction metrics (if multitask)
        if self.model.multitask:
            dir_metrics = self.val_dir_metrics.compute()
            self.log_dict(dir_metrics, prog_bar=False)
            self.val_dir_metrics.reset()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                }
            }
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        else:
            return optimizer


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [N, ...]
            targets: Target labels [N, ...] (same shape as inputs)
            
        Returns:
            Focal loss
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Compute alpha term
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SynfulTrainer:
    """
    High-level trainer for Synful models.
    
    Provides easy-to-use interface for training with best practices
    and automatic hyperparameter optimization.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "./synful_experiments",
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration dict
            data_config: Data configuration dict  
            training_config: Training configuration dict
            output_dir: Directory for experiment outputs
            experiment_name: Name for this experiment
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config or {}
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or "synful_experiment"
        self.use_wandb = use_wandb
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup model
        model_kwargs = dict(model_config)
        # Map dropout_rate to dropout if present
        if 'dropout_rate' in model_kwargs:
            model_kwargs['dropout'] = model_kwargs.pop('dropout_rate')
        # Remove unsupported parameters
        supported_params = {
            'n_channels', 'n_classes_mask', 'n_classes_vector', 'base_features',
            'depth', 'downsample_factors', 'multitask', 'bilinear', 'dropout', 'activation'
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in supported_params}
        
        self.model = UNet3D(**model_kwargs)
        self.lightning_module = None
        self.trainer = None
    
    def setup_lightning_module(self) -> SynfulLightningModule:
        """Setup Lightning module with training configuration."""
        if self.lightning_module is None:
            # Filter training config to only include parameters that SynfulLightningModule accepts
            supported_params = {
                'learning_rate', 'weight_decay', 'scheduler', 'mask_loss_weight', 
                'direction_loss_weight', 'focal_alpha', 'focal_gamma', 'augment_prob'
            }
            
            lightning_kwargs = {}
            for key in supported_params:
                if key in self.training_config:
                    lightning_kwargs[key] = self.training_config[key]
            
            # Map some parameter names if needed
            if 'loss_weights' in self.training_config:
                loss_weights = self.training_config['loss_weights']
                if 'mask' in loss_weights:
                    lightning_kwargs['mask_loss_weight'] = loss_weights['mask']
                if 'direction' in loss_weights:
                    lightning_kwargs['direction_loss_weight'] = loss_weights['direction']
            
            self.lightning_module = SynfulLightningModule(
                model=self.model,
                **lightning_kwargs
            )
        return self.lightning_module
    
    def setup_trainer(
        self,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: str = "auto",
        precision: str = "16-mixed",
        find_unused_parameters: bool = False,
    ) -> L.Trainer:
        """
        Setup PyTorch Lightning trainer.
        
        Args:
            max_epochs: Maximum number of training epochs
            accelerator: Accelerator type ("auto", "gpu", "cpu")
            devices: Number/list of devices to use
            precision: Training precision ("32", "16-mixed", "bf16-mixed")
            find_unused_parameters: For DDP with unused parameters
            
        Returns:
            Configured Lightning trainer
        """
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename=f"{self.experiment_name}-{{epoch:02d}}-{{val/total_loss:.4f}}",
                monitor="val/total_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val/total_loss",
                mode="min", 
                patience=20,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ModelSummary(max_depth=3),
        ]
        
        # Add snapshot callback if save_every is specified
        save_every = self.training_config.get('save_every')
        if save_every and save_every > 0:
            snapshot_callback = SnapshotCallback(
                snapshot_dir=self.output_dir / "snapshots",
                save_every=save_every,
                experiment_name=self.experiment_name,
                verbose=True
            )
            callbacks.append(snapshot_callback)
            logger.info(f"📸 Snapshot saving enabled: every {save_every} steps")
        
        # Setup loggers
        loggers = []
        
        # Weights & Biases logger
        if self.use_wandb:
            try:
                wandb_logger = WandbLogger(
                    project="synful-pytorch",
                    name=self.experiment_name,
                    save_dir=self.output_dir,
                    config={
                        **self.model_config,
                        **self.data_config, 
                        **self.training_config,
                    }
                )
                loggers.append(wandb_logger)
            except ImportError:
                logger.warning("Weights & Biases not available. Install with: pip install wandb")
        
        # Create trainer
        self.trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            logger=loggers,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        return self.trainer
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        **trainer_kwargs
    ):
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            **trainer_kwargs: Additional trainer arguments
        """
        # Setup Lightning module and trainer
        lightning_module = self.setup_lightning_module()
        trainer = self.setup_trainer(**trainer_kwargs)
        
        # Extract resume checkpoint if provided
        resume_from_checkpoint = trainer_kwargs.get('resume_from_checkpoint', None)
        
        # Start training
        logger.info(f"Starting training for experiment: {self.experiment_name}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        
        trainer.fit(
            lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=resume_from_checkpoint
        )
        
        logger.info("Training completed!")
        
        # Save final model
        final_model_path = self.output_dir / f"{self.experiment_name}_final.ckpt"
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Create snapshot summary
        self._create_snapshot_summary()
        
        logger.info("✅ Training completed successfully!")
    
    def find_learning_rate(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Find optimal learning rate using Lightning's LR finder.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Suggested learning rate
        """
        lightning_module = self.setup_lightning_module()
        trainer = self.setup_trainer(max_epochs=1)
        
        # Run LR finder
        lr_finder = trainer.tuner.lr_find(
            lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Get suggestion
        suggested_lr = lr_finder.suggestion()
        logger.info(f"Suggested learning rate: {suggested_lr}")
        
        # Plot results
        fig = lr_finder.plot(suggest=True)
        fig.savefig(self.output_dir / "lr_finder.png")
        
        return suggested_lr
    
    def _create_snapshot_summary(self):
        """Create a summary of all saved snapshots."""
        snapshot_dir = self.output_dir / "snapshots"
        if not snapshot_dir.exists():
            return
            
        snapshot_manager = SnapshotManager(snapshot_dir)
        snapshots = snapshot_manager.list_snapshots()
        
        if not snapshots:
            logger.info("📸 No snapshots found")
            return
        
        logger.info(f"📸 Snapshot Summary ({len(snapshots)} snapshots):")
        for snapshot in snapshots:
            logger.info(f"   {snapshot['filename']}: Step {snapshot['step']}, "
                       f"Epoch {snapshot['epoch']}, {snapshot['size_mb']:.1f}MB")
        
        # Save summary to file
        summary_file = snapshot_dir / "snapshot_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Snapshot Summary for {self.experiment_name}\n")
            f.write("="*50 + "\n\n")
            
            for snapshot in snapshots:
                f.write(f"File: {snapshot['filename']}\n")
                f.write(f"  Step: {snapshot['step']}\n")
                f.write(f"  Epoch: {snapshot['epoch']}\n")
                f.write(f"  Size: {snapshot['size_mb']:.1f}MB\n")
                f.write(f"  Path: {snapshot['path']}\n\n")
        
        logger.info(f"📄 Snapshot summary saved to: {summary_file}")
    
    def resume_from_snapshot(self, snapshot_path: str):
        """
        Resume training from a snapshot.
        
        Args:
            snapshot_path: Path to snapshot file or 'latest' for most recent
        """
        snapshot_dir = self.output_dir / "snapshots"
        snapshot_manager = SnapshotManager(snapshot_dir)
        
        if snapshot_path == 'latest':
            snapshot_file = snapshot_manager.find_latest_snapshot()
            if snapshot_file is None:
                raise FileNotFoundError("No snapshots found to resume from")
        else:
            snapshot_file = Path(snapshot_path)
            
        logger.info(f"🔄 Resuming training from snapshot: {snapshot_file}")
        
        # The actual resume logic will be handled by Lightning's resume_from_checkpoint
        return str(snapshot_file)
    
    def list_snapshots(self):
        """List all available snapshots."""
        snapshot_dir = self.output_dir / "snapshots"
        if not snapshot_dir.exists():
            logger.info("📸 No snapshot directory found")
            return []
            
        snapshot_manager = SnapshotManager(snapshot_dir)
        snapshots = snapshot_manager.list_snapshots()
        
        if not snapshots:
            logger.info("📸 No snapshots found")
            return []
        
        logger.info(f"📸 Available snapshots ({len(snapshots)}):")
        for snapshot in snapshots:
            logger.info(f"   {snapshot['filename']}: Step {snapshot['step']}, "
                       f"Epoch {snapshot['epoch']}")
        
        return snapshots
    
    def cleanup_old_snapshots(self, keep_last: int = 5):
        """Clean up old snapshots, keeping only the most recent ones."""
        snapshot_dir = self.output_dir / "snapshots"
        if not snapshot_dir.exists():
            logger.info("📸 No snapshot directory found")
            return
            
        snapshot_manager = SnapshotManager(snapshot_dir)
        snapshot_manager.cleanup_old_snapshots(keep_last=keep_last)


def create_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Create default configuration dictionaries.
    
    Returns:
        Tuple of (model_config, data_config, training_config)
    """
    model_config = {
        "n_channels": 1,
        "base_features": 16,
        "depth": 4,
        "multitask": True,
        "dropout": 0.1,
        "activation": "ReLU",
    }
    
    data_config = {
        "batch_size": (32, 256, 256),
        "voxel_size": (8.0, 8.0, 8.0),
        "num_workers": 4,
        "mask_radius": 40.0,
        "direction_radius": 80.0,
    }
    
    training_config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "mask_loss_weight": 1.0,
        "direction_loss_weight": 0.5,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "augment_prob": 0.8,
    }
    
    return model_config, data_config, training_config