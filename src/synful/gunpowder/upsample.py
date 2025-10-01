"""
Modernized UpSample for Synful pipeline.

This node upsamples arrays by given factors using scipy interpolation.
"""

import logging
import numbers
from typing import Union, Tuple

import numpy as np
from scipy import ndimage

import gunpowder as gp
from gunpowder import BatchFilter, ArrayKey, Array, ArraySpec

logger = logging.getLogger(__name__)


class UpSample(BatchFilter):
    """
    Upsample arrays in a batch by given factors.

    Args:
        source (ArrayKey): The key of the array to upsample.
        
        factor (Union[int, Tuple[int, ...]]): The factor to upsample with.
        
        target (ArrayKey): The key of the array to store the upsampled source.
    """

    def __init__(
        self, 
        source: ArrayKey, 
        factor: Union[int, Tuple[int, ...]], 
        target: ArrayKey
    ):
        assert isinstance(source, ArrayKey)
        assert isinstance(target, ArrayKey)
        assert (isinstance(factor, numbers.Number) or isinstance(factor, tuple)), \
            "Scaling factor should be a number or a tuple of numbers."

        self.source = source
        self.factor = factor
        self.target = target

    def setup(self):
        """Setup the upsampling by providing the target array specification."""
        spec = self.spec[self.source].copy()
        spec.voxel_size /= self.factor
        self.provides(self.target, spec)

    def prepare(self, request: gp.BatchRequest):
        """Prepare the request by expanding source ROI to accommodate upsampling."""
        if self.target not in request:
            return

        logger.debug(f"preparing upsampling of {self.source}")

        request_roi = request[self.target].roi
        voxel_size = self.spec[self.source].voxel_size
        request_roi = request_roi.snap_to_grid(voxel_size, mode='grow')
        logger.debug(f"request ROI is {request_roi}")

        # Add or merge to batch request
        if self.source in request:
            request[self.source].roi = request[self.source].roi.union(request_roi)
            logger.debug(f"merging with existing request to {request[self.source].roi}")
        else:
            request[self.source] = ArraySpec(roi=request_roi)
            logger.debug("adding as new request")

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """Process the batch by upsampling the source array."""
        if self.target not in request:
            return

        input_roi = batch.arrays[self.source].spec.roi
        request_roi = request[self.target].roi

        assert input_roi.contains(request_roi), \
            f"Input ROI {input_roi} must contain request ROI {request_roi}"

        # Upsample using scipy
        # Use order=3 (cubic) for interpolatable arrays, order=0 (nearest) for discrete
        order = 3 if batch.arrays[self.source].spec.interpolatable else 0
        data = ndimage.zoom(
            batch.arrays[self.source].data, 
            np.array(self.factor), 
            order=order
        )

        # Create output array and crop accordingly
        spec = self.spec[self.target].copy()
        spec.roi = input_roi
        ar = Array(data, spec)
        batch.arrays[self.target] = ar.crop(request_roi)

        # Restore requested ROIs if needed
        if self.source in request:
            request_roi_source = request[self.source].roi

            if input_roi != request_roi_source:
                assert input_roi.contains(request_roi_source)
                logger.debug(
                    f"restoring original request roi {request_roi_source} "
                    f"of {self.source} from {input_roi}"
                )
                cropped = batch.arrays[self.source].crop(request_roi_source)
                batch.arrays[self.source] = cropped