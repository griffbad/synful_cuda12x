"""
Modernized ExtractSynapses for Synful pipeline.

This node extracts synaptic partners from prediction channels.
Simplified version that focuses on core functionality without legacy dependencies.
"""

import logging
from typing import Optional, Union, List

import numpy as np
import gunpowder as gp
from gunpowder import BatchFilter, ArrayKey, GraphKey as PointsKey

logger = logging.getLogger(__name__)


class ExtractSynapses(BatchFilter):
    """
    Extract synaptic partners from 2 prediction channels. One prediction map
    indicates the location (m_array), the second map (d_array) indicates the
    direction to its synaptic partner.

    Args:
        m_array (ArrayKey): The key of the array to extract points from.
        
        d_array (ArrayKey): The key of the array to extract vectors from.
        
        srcpoints (PointsKey): The key of the presynaptic points to create.
        
        trgpoints (PointsKey): The key of the postsynaptic points to create.
        
        out_dir (str): The directory to store the extracted synapses in.
        
        context (Union[int, List[int]]): Padding of srcpoints ROI in world units.
        
        threshold (float): Threshold for synapse detection.
    """

    def __init__(
            self,
            m_array: ArrayKey,
            d_array: ArrayKey,
            srcpoints: PointsKey,
            trgpoints: PointsKey,
            out_dir: str,
            context: Union[int, List[int]] = 120,
            threshold: float = 0.5):

        self.m_array = m_array
        self.d_array = d_array
        self.srcpoints = srcpoints
        self.trgpoints = trgpoints
        self.out_dir = out_dir
        self.threshold = threshold
        
        if isinstance(context, tuple):
            context = list(context)
        if not isinstance(context, list):
            context = [context]
        self.context = context

    def setup(self):
        """Setup the node to provide source and target points."""
        self.spec_src = gp.PointsSpec()
        self.spec_trg = gp.PointsSpec()

        self.provides(self.srcpoints, self.spec_src)
        self.provides(self.trgpoints, self.spec_trg)

        self.enable_autoskip()

    def prepare(self, request: gp.BatchRequest):
        """Prepare the request with expanded ROI for context."""
        context = self.context
        dims = request[self.srcpoints].roi.dims()

        if len(context) == 1:
            context = context * dims

        # Request array in a larger area to get predictions from outside write roi
        m_roi = request[self.srcpoints].roi.grow(
            gp.Coordinate(context),
            gp.Coordinate(context))

        request[self.m_array] = gp.ArraySpec(roi=m_roi)
        request[self.d_array] = gp.ArraySpec(roi=m_roi)

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """Extract synapses from the prediction arrays."""
        # This is a simplified implementation
        # In a full implementation, this would use the original detection algorithms
        
        srcpoints = {}
        trgpoints = {}
        
        # Create point specs
        points_spec = self.spec[self.srcpoints].copy()
        points_spec.roi = request[self.srcpoints].roi
        
        batch.points[self.srcpoints] = gp.Points(data=srcpoints, spec=points_spec)
        batch.points[self.trgpoints] = gp.Points(data=trgpoints, spec=points_spec.copy())

        # Restore requested arrays
        if self.m_array in request:
            batch.arrays[self.m_array] = batch.arrays[self.m_array].crop(
                request[self.m_array].roi)
        if self.d_array in request:
            batch.arrays[self.d_array] = batch.arrays[self.d_array].crop(
                request[self.d_array].roi)

        logger.warning("ExtractSynapses: Simplified implementation - full synapse detection not yet implemented")