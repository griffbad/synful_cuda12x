"""
Data sources for Synful PyTorch.

Provides specialized data sources for electron microscopy volumes
and synaptic annotations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import h5py
import zarr
from gunpowder import (
    GraphKey, GraphSpec, Graph, Node, Edge,
    BatchProvider, BatchRequest, Batch, Coordinate
)

# Conditional import to avoid circular dependency
try:
    from ..synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError:
    _SYNAPSE_AVAILABLE = False
    # Create placeholder classes if synapse module not available
    class Synapse:
        pass
    class SynapseCollection:
        pass

logger = logging.getLogger(__name__)


class HDF5SynapseSource(BatchProvider):
    """
    Gunpowder source for synapse annotations from HDF5 files.
    
    Expected HDF5 structure:
    - synapses/pre_locations: (N, 3) array of presynaptic locations
    - synapses/post_locations: (N, 3) array of postsynaptic locations
    - synapses/scores: (N,) array of detection scores
    - synapses/confidences: (N,) array of confidence values (optional)
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        synapses_key: GraphKey,
        pre_dataset: str = "synapses/pre_locations",
        post_dataset: str = "synapses/post_locations", 
        scores_dataset: str = "synapses/scores",
        confidences_dataset: str = "synapses/confidences",
    ):
        """
        Initialize HDF5 synapse source.
        
        Args:
            filename: Path to HDF5 file
            synapses_key: GraphKey for synapse graph
            pre_dataset: Dataset name for presynaptic locations
            post_dataset: Dataset name for postsynaptic locations
            scores_dataset: Dataset name for detection scores
            confidences_dataset: Dataset name for confidence values
        """
        self.filename = Path(filename)
        self.synapses_key = synapses_key
        self.pre_dataset = pre_dataset
        self.post_dataset = post_dataset
        self.scores_dataset = scores_dataset
        self.confidences_dataset = confidences_dataset
        
        self._synapses = None
        self._load_synapses()
    
    def _load_synapses(self):
        """Load synapses from HDF5 file."""
        try:
            with h5py.File(self.filename, 'r') as f:
                pre_locations = f[self.pre_dataset][:]
                post_locations = f[self.post_dataset][:]
                scores = f[self.scores_dataset][:]
                
                # Load confidences if available
                if self.confidences_dataset in f:
                    confidences = f[self.confidences_dataset][:]
                else:
                    confidences = np.ones_like(scores)
                
                # Create synapse collection
                synapses = []
                for i, (pre_loc, post_loc, score, conf) in enumerate(
                    zip(pre_locations, post_locations, scores, confidences)
                ):
                    synapse = Synapse(
                        id=i,
                        location_pre=tuple(pre_loc),
                        location_post=tuple(post_loc),
                        score=float(score),
                        confidence=float(conf)
                    )
                    synapses.append(synapse)
                
                self._synapses = SynapseCollection(synapses)
                logger.info(f"Loaded {len(synapses)} synapses from {self.filename}")
                
        except Exception as e:
            logger.error(f"Failed to load synapses from {self.filename}: {e}")
            self._synapses = SynapseCollection([])
    
    def setup(self):
        """Setup the synapse source."""
        # Calculate bounding box of all synapses
        if len(self._synapses) > 0:
            locations = []
            for synapse in self._synapses:
                locations.extend([synapse.location_pre, synapse.location_post])
            
            locations = np.array(locations)
            min_coords = locations.min(axis=0)
            max_coords = locations.max(axis=0)
            
            # Add some padding
            padding = 1000.0  # nm
            roi_begin = min_coords - padding
            roi_shape = max_coords - min_coords + 2 * padding
            
        else:
            # Empty dataset
            roi_begin = np.array([0.0, 0.0, 0.0])
            roi_shape = np.array([1.0, 1.0, 1.0])
        
        from gunpowder import Roi, Coordinate
        
        self.provides(
            self.synapses_key,
            GraphSpec(
                roi=Roi(Coordinate(roi_begin), Coordinate(roi_shape)),
                directed=True
            )
        )
    
    def provide(self, request):
        """Provide synapse data for the requested region."""
        from gunpowder import Graph, Node, Edge
        
        # Create graph
        graph = Graph([], [], GraphSpec(directed=True))
        
        # Add synapses that overlap with requested ROI
        synapse_roi = request[self.synapses_key].roi
        
        node_id = 0
        edge_id = 0
        
        for synapse in self._synapses:
            # Check if synapse locations are in ROI
            pre_in_roi = synapse_roi.contains(Coordinate(synapse.location_pre))
            post_in_roi = synapse_roi.contains(Coordinate(synapse.location_post))
            
            if pre_in_roi or post_in_roi:
                # Add presynaptic node
                pre_node = Node(
                    id=node_id,
                    location=Coordinate(synapse.location_pre),
                    synapse=synapse
                )
                graph.add_node(pre_node)
                pre_node_id = node_id
                node_id += 1
                
                # Add postsynaptic node
                post_node = Node(
                    id=node_id,
                    location=Coordinate(synapse.location_post),
                    synapse=synapse
                )
                graph.add_node(post_node)
                post_node_id = node_id
                node_id += 1
                
                # Add edge from pre to post
                edge = Edge(
                    id=edge_id,
                    u=pre_node_id,
                    v=post_node_id,
                    synapse=synapse
                )
                graph.add_edge(edge)
                edge_id += 1
        
        batch = Batch()
        batch[self.synapses_key] = graph
        return batch


class ZarrVolumeSource:
    """
    Modern Zarr volume source with cloud storage support.
    
    Provides efficient access to large EM volumes stored in Zarr format,
    with support for chunked reading and cloud storage backends.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        dataset: str = "volumes/raw",
        chunk_size: Optional[tuple] = None,
    ):
        """
        Initialize Zarr volume source.
        
        Args:
            path: Path to Zarr store (local or cloud URL)
            dataset: Dataset name within Zarr store
            chunk_size: Preferred chunk size for reading
        """
        self.path = str(path)
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        # Open Zarr array
        self._array = None
        self._open_array()
    
    def _open_array(self):
        """Open the Zarr array."""
        try:
            store = zarr.open(self.path, mode='r')
            self._array = store[self.dataset]
            logger.info(
                f"Opened Zarr array: {self._array.shape} "
                f"chunks={self._array.chunks} dtype={self._array.dtype}"
            )
        except Exception as e:
            logger.error(f"Failed to open Zarr array {self.path}:{self.dataset}: {e}")
            raise
    
    @property
    def shape(self) -> tuple:
        """Get array shape."""
        return self._array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get array dtype."""
        return self._array.dtype
    
    @property
    def chunks(self) -> tuple:
        """Get array chunk size."""
        return self._array.chunks
    
    def read_region(
        self,
        offset: tuple,
        shape: tuple
    ) -> np.ndarray:
        """
        Read a region from the volume.
        
        Args:
            offset: Starting coordinates (z, y, x)
            shape: Size of region to read (z, y, x)
            
        Returns:
            Array data for the requested region
        """
        slices = tuple(
            slice(off, off + sh)
            for off, sh in zip(offset, shape)
        )
        return self._array[slices]


class CloudVolumeSource:
    """
    Source for cloud-stored EM volumes using CloudVolume.
    
    Provides access to neuroglancer precomputed format volumes
    stored in cloud storage (GCS, S3, etc.).
    """
    
    def __init__(
        self,
        cloudpath: str,
        mip: int = 0,
        cache: bool = True,
    ):
        """
        Initialize cloud volume source.
        
        Args:
            cloudpath: CloudVolume path (e.g., 'gs://bucket/path')
            mip: MIP level to read (0 = highest resolution)
            cache: Whether to enable local caching
        """
        self.cloudpath = cloudpath
        self.mip = mip
        self.cache = cache
        
        # Import CloudVolume
        try:
            from cloudvolume import CloudVolume
            self._cv = CloudVolume(
                cloudpath,
                mip=mip,
                cache=cache,
                parallel=True
            )
            logger.info(
                f"Opened CloudVolume: {self._cv.shape} "
                f"voxel_size={self._cv.resolution} dtype={self._cv.dtype}"
            )
        except ImportError:
            logger.error("CloudVolume not available. Install with: pip install cloud-volume")
            raise
        except Exception as e:
            logger.error(f"Failed to open CloudVolume {cloudpath}: {e}")
            raise
    
    @property
    def shape(self) -> tuple:
        """Get volume shape."""
        return self._cv.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get volume dtype.""" 
        return self._cv.dtype
    
    @property
    def resolution(self) -> tuple:
        """Get voxel resolution in nm."""
        return self._cv.resolution
    
    def read_region(
        self,
        offset: tuple,
        shape: tuple
    ) -> np.ndarray:
        """
        Read a region from the cloud volume.
        
        Args:
            offset: Starting coordinates (x, y, z) in CloudVolume order
            shape: Size of region to read (x, y, z) in CloudVolume order
            
        Returns:
            Array data for the requested region
        """
        # CloudVolume uses (x, y, z) order
        end_coords = tuple(off + sh for off, sh in zip(offset, shape))
        bbox = slice(*offset), slice(*end_coords)
        
        data = self._cv[bbox]
        
        # Convert from (x, y, z, c) to (z, y, x) if needed
        if len(data.shape) == 4:
            data = data[..., 0]  # Remove channel dimension
        
        # Transpose from (x, y, z) to (z, y, x)
        data = np.transpose(data, (2, 1, 0))
        
        return data