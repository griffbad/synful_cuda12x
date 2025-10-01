"""
Modern PyTorch implementation of U-Net for synaptic partner detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import lightning as L
from torchmetrics import MetricCollection, JaccardIndex
try:
    from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
    F1Score = BinaryF1Score
    JaccardIndex = BinaryJaccardIndex
except ImportError:
    try:
        from torchmetrics import F1Score
    except ImportError:
        # Fallback for very old versions
        from torchmetrics.functional import f1_score as F1Score
import numpy as np


class ConvBlock3D(nn.Module):
    """3D Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation: str = "ReLU",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        
        # Convolution
        layers.append(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=padding,
                bias=not use_batchnorm
            )
        )
        
        # Batch normalization
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
            
        # Activation
        if activation == "ReLU":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "LeakyReLU":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "GELU":
            layers.append(nn.GELU())
            
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv3D(nn.Module):
    """Double 3D convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        self.conv1 = ConvBlock3D(in_channels, mid_channels, **kwargs)
        self.conv2 = ConvBlock3D(mid_channels, out_channels, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down3D(nn.Module):
    """Downsampling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, pool_size: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pool_size),
            DoubleConv3D(in_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upsampling then double conv."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        scale_factor: Tuple[int, int, int] = (2, 2, 2),
        bilinear: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
            # For concatenation: in_channels (from skip) + in_channels//2 (from upsampling)
            self.conv = DoubleConv3D(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            # For concatenation: in_channels (from skip) + in_channels//2 (from upsampling)  
            self.conv = DoubleConv3D(in_channels, out_channels)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle potential size mismatches
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Modern 3D U-Net implementation for synaptic partner detection.
    
    This implementation supports both single-task (mask only) and multi-task
    (mask + direction vectors) prediction.
    """
    
    def __init__(
        self,
        n_channels: int = 1,
        n_classes_mask: int = 1,
        n_classes_vector: int = 3,
        base_features: int = 64,
        depth: int = 4,
        downsample_factors: List[Tuple[int, int, int]] = None,
        multitask: bool = True,
        bilinear: bool = True,
        dropout: float = 0.1,
        activation: str = "ReLU",
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes_mask = n_classes_mask
        self.n_classes_vector = n_classes_vector
        self.multitask = multitask
        self.depth = depth
        
        if downsample_factors is None:
            downsample_factors = [(2, 2, 2)] * depth
            
        # Input convolution
        self.inc = DoubleConv3D(n_channels, base_features, dropout=dropout, activation=activation)
        
        # Encoder
        self.downs = nn.ModuleList()
        in_features = base_features
        for i in range(depth):
            out_features = base_features * (2 ** (i + 1))
            self.downs.append(Down3D(in_features, out_features, downsample_factors[i]))
            in_features = out_features
            
        # Decoder for mask
        self.ups_mask = nn.ModuleList()
        for i in range(depth):
            # Calculate input channels: current features + skip connection features
            skip_channels = base_features * (2 ** (depth - i - 1))  # From encoder
            up_channels = base_features * (2 ** (depth - i))        # From previous decoder
            total_in_channels = skip_channels + up_channels // 2    # After upsampling
            out_features = base_features * (2 ** (depth - i - 1))
            
            self.ups_mask.append(Up3D(up_channels, out_features, downsample_factors[depth - i - 1], bilinear))
            
        # Output for mask
        self.outc_mask = nn.Conv3d(base_features, n_classes_mask, kernel_size=1)
        
        # Decoder for direction vectors (if multitask)
        if multitask:
            self.ups_vector = nn.ModuleList()
            for i in range(depth):
                # Same calculation for vector decoder
                skip_channels = base_features * (2 ** (depth - i - 1))
                up_channels = base_features * (2 ** (depth - i))
                total_in_channels = skip_channels + up_channels // 2
                out_features = base_features * (2 ** (depth - i - 1))
                
                self.ups_vector.append(Up3D(up_channels, out_features, downsample_factors[depth - i - 1], bilinear))
                
            # Output for direction vectors
            self.outc_vector = nn.Conv3d(base_features, n_classes_vector, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        x1 = self.inc(x)
        
        encoder_features = [x1]
        x = x1
        for down in self.downs:
            x = down(x)
            encoder_features.append(x)
            
        # Decoder for mask
        x_mask = encoder_features[-1]
        for i, up in enumerate(self.ups_mask):
            x_mask = up(x_mask, encoder_features[-(i + 2)])
            
        mask_logits = self.outc_mask(x_mask)
        
        outputs = {"mask_logits": mask_logits}
        
        # Decoder for direction vectors (if multitask)
        if self.multitask:
            x_vector = encoder_features[-1]
            for i, up in enumerate(self.ups_vector):
                x_vector = up(x_vector, encoder_features[-(i + 2)])
                
            vector_logits = self.outc_vector(x_vector)
            outputs["direction_vectors"] = vector_logits
            
        return outputs


class SynfulModel(L.LightningModule):
    """
    Lightning module for Synful training and inference.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = UNet3D(**model_config)
        
        # Loss configuration
        self.loss_config = loss_config
        self.mask_loss_weight = loss_config.get("mask_loss_weight", 1.0)
        self.vector_loss_weight = loss_config.get("vector_loss_weight", 1.0)
        
        # Metrics
        try:
            self.train_metrics = MetricCollection({
                "f1": F1Score(task="binary"),
                "iou": JaccardIndex(task="binary"),
            })
        except Exception:
            # Fallback for compatibility
            self.train_metrics = MetricCollection({
                "f1": F1Score(),
                "iou": JaccardIndex(task="binary", num_classes=2),
            })
        
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined loss for mask and vector predictions."""
        total_loss = 0.0
        
        # Mask loss
        if "mask_logits" in outputs and "mask" in targets:
            mask_loss = F.binary_cross_entropy_with_logits(
                outputs["mask_logits"], targets["mask"]
            )
            total_loss += self.mask_loss_weight * mask_loss
            
        # Vector loss
        if "vector_logits" in outputs and "vectors" in targets:
            vector_loss = F.mse_loss(outputs["vector_logits"], targets["vectors"])
            total_loss += self.vector_loss_weight * vector_loss
            
        return total_loss
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["image"])
        loss = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute and log additional metrics
        if "mask_logits" in outputs and "mask" in batch:
            mask_pred = torch.sigmoid(outputs["mask_logits"])
            mask_pred_binary = (mask_pred > 0.5).int()
            try:
                metrics = self.train_metrics(mask_pred_binary, batch["mask"].int())
                self.log_dict(metrics, on_step=False, on_epoch=True)
            except Exception as e:
                # Skip metrics if there's an error
                self.log("train_metrics_error", 1.0, on_step=False, on_epoch=True)
            
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["image"])
        loss = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute and log additional metrics
        if "mask_logits" in outputs and "mask" in batch:
            mask_pred = torch.sigmoid(outputs["mask_logits"])
            mask_pred_binary = (mask_pred > 0.5).int()
            try:
                metrics = self.val_metrics(mask_pred_binary, batch["mask"].int())
                self.log_dict(metrics, on_step=False, on_epoch=True)
            except Exception as e:
                # Skip metrics if there's an error
                self.log("val_metrics_error", 1.0, on_step=False, on_epoch=True)
            
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.hparams.optimizer_config
        )
        
        if self.hparams.scheduler_config is None:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **self.hparams.scheduler_config
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }