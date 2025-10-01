"""
Snapshot callback for Synful training.

Provides periodic model snapshots during training, similar to the original
Synful implementation. This is useful for:
- Long training runs where you want regular checkpoints
- Recovery from training interruptions  
- Analysis of training progression
- Creating ensembles from different training stages
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class SnapshotCallback(Callback):
    """
    Callback to save model snapshots at regular intervals.
    
    Similar to the original Synful snapshot feature, this saves the model
    state at fixed iteration intervals, providing regular checkpoints
    throughout training.
    """
    
    def __init__(
        self,
        snapshot_dir: str,
        save_every: int = 30000,
        experiment_name: str = "synful",
        save_optimizer: bool = True,
        save_lr_scheduler: bool = True,
        verbose: bool = True
    ):
        """
        Initialize snapshot callback.
        
        Args:
            snapshot_dir: Directory to save snapshots
            save_every: Save snapshot every N training steps
            experiment_name: Prefix for snapshot filenames
            save_optimizer: Whether to save optimizer state
            save_lr_scheduler: Whether to save LR scheduler state
            verbose: Whether to print snapshot messages
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.save_every = save_every
        self.experiment_name = experiment_name
        self.save_optimizer = save_optimizer
        self.save_lr_scheduler = save_lr_scheduler
        self.verbose = verbose
        
        # Create snapshot directory
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Track steps
        self.global_step = 0
        
    @rank_zero_only
    def on_train_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx: int
    ) -> None:
        """Called after each training batch."""
        self.global_step = trainer.global_step
        
        # Check if it's time to save snapshot
        if self.global_step > 0 and self.global_step % self.save_every == 0:
            self._save_snapshot(trainer, pl_module)
    
    def _save_snapshot(self, trainer, pl_module) -> None:
        """Save a snapshot of the current training state."""
        snapshot_path = self.snapshot_dir / f"{self.experiment_name}_step_{self.global_step:07d}.ckpt"
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': self.global_step,
            'pytorch-lightning_version': trainer.lightning_module.__class__.__module__.split('.')[0],
            'state_dict': pl_module.state_dict(),
            'lr_schedulers': [],
            'optimizer_states': [],
        }
        
        # Add optimizer states
        if self.save_optimizer:
            for optimizer in trainer.optimizers:
                checkpoint['optimizer_states'].append(optimizer.state_dict())
        
        # Add LR scheduler states
        if self.save_lr_scheduler:
            for lr_scheduler in trainer.lr_scheduler_configs:
                checkpoint['lr_schedulers'].append(lr_scheduler.scheduler.state_dict())
        
        # Add hyperparameters
        if hasattr(pl_module, 'hparams'):
            checkpoint['hyper_parameters'] = pl_module.hparams
        
        # Save checkpoint
        try:
            torch.save(checkpoint, snapshot_path)
            
            if self.verbose:
                logger.info(f"💾 Snapshot saved: {snapshot_path}")
                logger.info(f"   Step: {self.global_step}, Epoch: {trainer.current_epoch}")
                
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    @rank_zero_only 
    def on_train_end(self, trainer, pl_module) -> None:
        """Save final snapshot at end of training."""
        final_path = self.snapshot_dir / f"{self.experiment_name}_final.ckpt"
        
        # Save final checkpoint
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': self.global_step,
            'pytorch-lightning_version': trainer.lightning_module.__class__.__module__.split('.')[0],
            'state_dict': pl_module.state_dict(),
            'lr_schedulers': [],
            'optimizer_states': [],
        }
        
        # Add optimizer and scheduler states
        if self.save_optimizer:
            for optimizer in trainer.optimizers:
                checkpoint['optimizer_states'].append(optimizer.state_dict())
        
        if self.save_lr_scheduler:
            for lr_scheduler in trainer.lr_scheduler_configs:
                checkpoint['lr_schedulers'].append(lr_scheduler.scheduler.state_dict())
        
        if hasattr(pl_module, 'hparams'):
            checkpoint['hyper_parameters'] = pl_module.hparams
        
        try:
            torch.save(checkpoint, final_path)
            if self.verbose:
                logger.info(f"💾 Final snapshot saved: {final_path}")
        except Exception as e:
            logger.error(f"Failed to save final snapshot: {e}")


class SnapshotManager:
    """
    Utility class for managing and loading snapshots.
    
    Provides functionality to:
    - List available snapshots
    - Load specific snapshots
    - Resume training from snapshots
    - Clean up old snapshots
    """
    
    def __init__(self, snapshot_dir: str):
        """Initialize snapshot manager."""
        self.snapshot_dir = Path(snapshot_dir)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots with metadata."""
        snapshots = []
        
        for ckpt_file in self.snapshot_dir.glob("*.ckpt"):
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                
                snapshot_info = {
                    'path': ckpt_file,
                    'filename': ckpt_file.name,
                    'step': checkpoint.get('global_step', 0),
                    'epoch': checkpoint.get('epoch', 0),
                    'size_mb': ckpt_file.stat().st_size / (1024 * 1024),
                    'modified': ckpt_file.stat().st_mtime,
                }
                
                snapshots.append(snapshot_info)
                
            except Exception as e:
                logger.warning(f"Could not read snapshot {ckpt_file}: {e}")
        
        # Sort by step
        snapshots.sort(key=lambda x: x['step'])
        return snapshots
    
    def load_snapshot(self, snapshot_path: str) -> Dict[str, Any]:
        """Load a specific snapshot."""
        snapshot_path = Path(snapshot_path)
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
        
        try:
            checkpoint = torch.load(snapshot_path, map_location='cpu')
            logger.info(f"📥 Loaded snapshot: {snapshot_path}")
            logger.info(f"   Step: {checkpoint.get('global_step', 'unknown')}")
            logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_path}: {e}")
            raise
    
    def find_latest_snapshot(self) -> Optional[Path]:
        """Find the latest snapshot by step number."""
        snapshots = self.list_snapshots()
        
        if not snapshots:
            return None
        
        # Return latest by step
        latest = max(snapshots, key=lambda x: x['step'])
        return latest['path']
    
    def cleanup_old_snapshots(self, keep_last: int = 5) -> None:
        """Remove old snapshots, keeping only the most recent ones."""
        snapshots = self.list_snapshots()
        
        if len(snapshots) <= keep_last:
            logger.info(f"Found {len(snapshots)} snapshots, no cleanup needed")
            return
        
        # Sort by step and remove oldest
        snapshots.sort(key=lambda x: x['step'])
        to_remove = snapshots[:-keep_last]
        
        for snapshot in to_remove:
            try:
                snapshot['path'].unlink()
                logger.info(f"🗑️  Removed old snapshot: {snapshot['filename']}")
            except Exception as e:
                logger.warning(f"Could not remove snapshot {snapshot['filename']}: {e}")
        
        logger.info(f"Cleaned up {len(to_remove)} old snapshots, kept {keep_last} most recent")


def resume_from_snapshot(
    snapshot_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Resume training from a snapshot.
    
    Args:
        snapshot_path: Path to snapshot file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        lr_scheduler: LR scheduler to load state into (optional)
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Dictionary with resume information (epoch, step, etc.)
    """
    manager = SnapshotManager(Path(snapshot_path).parent)
    checkpoint = manager.load_snapshot(snapshot_path)
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    logger.info("✅ Model state loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_states' in checkpoint:
        if checkpoint['optimizer_states']:
            optimizer.load_state_dict(checkpoint['optimizer_states'][0])
            logger.info("✅ Optimizer state loaded")
    
    # Load LR scheduler state
    if lr_scheduler is not None and 'lr_schedulers' in checkpoint:
        if checkpoint['lr_schedulers']:
            lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
            logger.info("✅ LR scheduler state loaded")
    
    resume_info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'checkpoint_path': snapshot_path
    }
    
    logger.info(f"🚀 Resuming from step {resume_info['global_step']}, epoch {resume_info['epoch']}")
    return resume_info