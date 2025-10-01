"""
Modernized IntensityScaleShiftClip for Synful pipeline.

This node provides intensity scaling, shifting, and optional clipping operations.
"""

from typing import Optional, Tuple

import numpy as np
import gunpowder as gp
from gunpowder import BatchFilter, ArrayKey


class IntensityScaleShiftClip(BatchFilter):
    """
    Scales the intensities of a batch by scale, then adds shift.
    Optionally, also clips arrays.

    Args:
        array (ArrayKey): The key of the array to modify.
        
        scale (float): The scale factor to apply to array.
        
        shift (float): The shift value to add to array.
        
        clip (Tuple[float, float], optional): Clip_min and clip_max value, 
            clipping applied after scaling.
    """

    def __init__(
        self, 
        array: ArrayKey, 
        scale: float, 
        shift: float, 
        clip: Optional[Tuple[float, float]] = None
    ):
        self.array = array
        self.scale = scale
        self.shift = shift
        self.clip = clip

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """Apply scale, shift, and optional clipping to the specified array."""
        if self.array not in batch.arrays:
            return

        raw = batch.arrays[self.array]
        raw.data = raw.data * self.scale + self.shift
        
        if self.clip is not None:
            raw.data = np.clip(raw.data, self.clip[0], self.clip[1])