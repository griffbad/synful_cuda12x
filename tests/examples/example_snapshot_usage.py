#!/usr/bin/env python3
"""
Example script demonstrating snapshot functionality in Synful training.

This script shows how to:
1. Enable snapshots during training
2. Resume from snapshots
3. List and manage snapshots
4. Clean up old snapshots
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_snapshot_usage():
    """Demonstrate snapshot functionality."""
    
    logger.info("🔍 Synful Snapshot Feature Demonstration")
    logger.info("=" * 50)
    
    print("""
    Synful training now supports snapshots! Here's how to use them:
    
    1. ENABLE SNAPSHOTS DURING TRAINING:
    ===================================
    
    cd scripts/train/setup03
    python train_pytorch.py parameter.json --snapshot-every 500 --keep-snapshots 10
    
    This will:
    - Save a snapshot every 500 training steps
    - Keep the 10 most recent snapshots
    - Automatically clean up older snapshots
    
    
    2. RESUME FROM A SNAPSHOT:
    =========================
    
    # Resume from the latest snapshot
    python train_pytorch.py parameter.json --resume-snapshot latest
    
    # Resume from a specific snapshot file
    python train_pytorch.py parameter.json --resume-snapshot ./snapshots/snapshot_step_2000.ckpt
    
    
    3. SNAPSHOT MANAGEMENT:
    ======================
    
    Snapshots are saved in: ./snapshots/
    
    Each snapshot contains:
    - Model weights and optimizer state
    - Training step and epoch information
    - All necessary state to resume training
    
    
    4. AUTOMATIC CLEANUP:
    ====================
    
    The system automatically:
    - Removes old snapshots when limit is reached
    - Creates a summary file with all snapshot info
    - Logs snapshot creation and cleanup activities
    
    
    5. FULL TRAINING EXAMPLE:
    ========================
    
    # Train with real data and snapshots
    python train_pytorch.py parameter.json \\
        --zarr-path /path/to/volume.zarr \\
        --tsv-synapses /path/to/synapses.tsv \\
        --snapshot-every 1000 \\
        --keep-snapshots 5 \\
        --epochs 100
    
    # If training is interrupted, resume with:
    python train_pytorch.py parameter.json \\
        --zarr-path /path/to/volume.zarr \\
        --tsv-synapses /path/to/synapses.tsv \\
        --resume-snapshot latest
    
    
    6. TSV FILE FORMAT FOR SYNAPSES:
    ================================
    
    The TSV file should have coordinates in this format:
    
    pre_x   pre_y   pre_z   post_x  post_y  post_z
    100     200     150     105     205     155
    200     300     250     195     295     245
    ...     ...     ...     ...     ...     ...
    
    No header row - just coordinates separated by tabs.
    Coordinates can be in xyz or zyx order (specify with coordinate_order).
    
    
    7. SYNTHETIC DATA FOR TESTING:
    ==============================
    
    # Test the snapshot system with synthetic data
    python train_pytorch.py parameter.json \\
        --synthetic \\
        --snapshot-every 100 \\
        --epochs 5
    
    This is perfect for testing the snapshot system without needing real data!
    """)
    
    logger.info("📚 For more details, see the training script:")
    logger.info("   scripts/train/setup03/train_pytorch.py --help")


if __name__ == "__main__":
    demonstrate_snapshot_usage()