"""
Modernized CloudVolumeSource for Synful pipeline.

This node provides access to CloudVolume data sources.
Note: CloudVolume dependency is optional and may need to be installed separately.
"""

import logging
from typing import Optional

import numpy as np
import gunpowder as gp
from gunpowder import Batch, BatchProvider, ArrayKey, ArraySpec, Coordinate, Roi, Array

logger = logging.getLogger(__name__)

try:
    from cloudvolume import CloudVolume
    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False
    logger.warning("CloudVolume not available. CloudVolumeSource will not work.")


class CloudVolumeSource(BatchProvider):
    """
    A source for CloudVolume.

    Provides array from CloudVolume.

    Args:
        cloudvolume_url (str): The cloudvolume url from which to load the array.
        
        array_key (ArrayKey): Array key for the array.
        
        mip (int): MIP level to use.
        
        array_spec (ArraySpec, optional): Array spec to overwrite the array spec 
            automatically determined from the cloudvolume meta information.
    """

    def __init__(
            self,
            cloudvolume_url: str,
            array_key: ArrayKey,
            mip: int = 0,
            array_spec: Optional[ArraySpec] = None):

        if not CLOUDVOLUME_AVAILABLE:
            raise ImportError("CloudVolume is required for CloudVolumeSource. Install with: pip install cloud-volume")
            
        self.cloudvolume_url = cloudvolume_url
        self.mip = mip
        self.array_key = array_key
        self.array_spec = array_spec
        self.ndims = None

    def setup(self):
        """Setup the provider by reading CloudVolume specifications."""
        cv = CloudVolume(self.cloudvolume_url, use_https=True, mip=self.mip)
        spec = self._read_spec(cv)
        logger.debug(f'Spec is {spec}')

        self.array_spec = spec
        self.provides(self.array_key, spec)

    def provide(self, request: gp.BatchRequest) -> Batch:
        """Provide data from CloudVolume according to the request."""
        batch = Batch()

        cv = CloudVolume(self.cloudvolume_url, use_https=True, mip=self.mip)

        request_spec = request.array_specs[self.array_key]
        array_key = self.array_key
        logger.debug(f"Reading {array_key} in {request_spec.roi}...")

        voxel_size = self.array_spec.voxel_size

        # Scale request roi to voxel units
        dataset_roi = request_spec.roi / voxel_size

        # Shift request roi into dataset
        dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

        # Create array spec
        array_spec = self.array_spec.copy()
        array_spec.roi = request_spec.roi

        # Add array to batch
        batch.arrays[array_key] = Array(
            self._read(cv, dataset_roi),
            array_spec)

        logger.debug("CloudVolume read completed")
        return batch

    def _read_spec(self, cv) -> ArraySpec:
        """Read array specification from CloudVolume."""
        info = cv.info
        
        # Get voxel size from info
        voxel_size = Coordinate(info['scales'][self.mip]['voxel_offset'])
        
        # Get shape and create ROI
        shape = Coordinate(info['scales'][self.mip]['size'])
        roi = Roi(Coordinate((0,) * len(shape)), shape * voxel_size)
        
        # Determine data type
        if info['data_type'] == 'uint8':
            dtype = np.uint8
        elif info['data_type'] == 'uint16':
            dtype = np.uint16
        elif info['data_type'] == 'float32':
            dtype = np.float32
        else:
            dtype = np.float32
            logger.warning(f"Unknown data type {info['data_type']}, using float32")

        # Override with user-provided spec if available
        if self.array_spec is not None:
            if self.array_spec.voxel_size is not None:
                voxel_size = self.array_spec.voxel_size
            if self.array_spec.dtype is not None:
                dtype = self.array_spec.dtype
            if self.array_spec.roi is not None:
                roi = self.array_spec.roi

        return ArraySpec(roi=roi, voxel_size=voxel_size, dtype=dtype)

    def _read(self, cv, dataset_roi: Roi) -> np.ndarray:
        """Read data from CloudVolume for the specified ROI."""
        begin = dataset_roi.get_begin()
        end = dataset_roi.get_end()
        
        # CloudVolume uses [x, y, z] order while we use [z, y, x]
        # Need to handle coordinate system conversion
        begin_cv = [int(begin[2]), int(begin[1]), int(begin[0])]
        end_cv = [int(end[2]), int(end[1]), int(end[0])]
        
        data = cv[begin_cv[0]:end_cv[0], begin_cv[1]:end_cv[1], begin_cv[2]:end_cv[2]]
        
        # Convert back to [z, y, x] order
        data = np.transpose(data, (2, 1, 0))
        
        return data

    def __repr__(self):
        return f"CloudVolumeSource({self.cloudvolume_url})"