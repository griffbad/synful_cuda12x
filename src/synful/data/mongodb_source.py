"""
MongoDB data source for Synful.

Provides integration with MongoDB databases containing synapse annotations,
compatible with the original Synful database schema.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from pymongo import MongoClient
from gunpowder import BatchProvider, GraphSpec, GraphKey, Graph, Node, Edge, Roi, Coordinate

# Conditional import to avoid circular dependency
try:
    from ..synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError:
    _SYNAPSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MongoDBSynapseSource(BatchProvider):
    """
    MongoDB source for synapse annotations.
    
    Compatible with the original Synful database schema where synapses are stored
    with pre/post locations and can be queried spatially.
    """
    
    def __init__(
        self,
        db_host: str,
        db_name: str,
        collection_name: str,
        synapses_key: GraphKey,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        roi: Optional[Roi] = None
    ):
        """
        Initialize MongoDB synapse source.
        
        Args:
            db_host: MongoDB host address
            db_name: Database name
            collection_name: Collection name (without .synapses suffix)
            synapses_key: GraphKey for synapse data
            voxel_size: Voxel size in nm (z, y, x)
            roi: Optional ROI to restrict queries to
        """
        self.db_host = db_host
        self.db_name = db_name
        self.collection_name = collection_name
        self.synapses_key = synapses_key
        self.voxel_size = Coordinate(voxel_size)
        self.roi = roi
        
        # Connect to MongoDB
        self._client = None
        self._collection = None
        self._connect()
        
    def _connect(self):
        """Establish MongoDB connection."""
        try:
            self._client = MongoClient(self.db_host)
            self._db = self._client[self.db_name]
            self._collection = self._db[f"{self.collection_name}.synapses"]
            logger.info(f"Connected to MongoDB: {self.db_host}/{self.db_name}/{self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
            
    def setup(self):
        """Setup the provider."""
        # If no ROI provided, estimate from database
        if self.roi is None:
            self.roi = self._estimate_roi()
            
        self.provides(
            self.synapses_key,
            GraphSpec(roi=self.roi, directed=True)
        )
        
    def _estimate_roi(self) -> Roi:
        """Estimate ROI from all synapses in database."""
        try:
            # Query min/max coordinates
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "min_pre_z": {"$min": "$pre_z"},
                        "max_pre_z": {"$max": "$pre_z"},
                        "min_pre_y": {"$min": "$pre_y"},
                        "max_pre_y": {"$max": "$pre_y"},
                        "min_pre_x": {"$min": "$pre_x"},
                        "max_pre_x": {"$max": "$pre_x"},
                        "min_post_z": {"$min": "$post_z"},
                        "max_post_z": {"$max": "$post_z"},
                        "min_post_y": {"$min": "$post_y"},
                        "max_post_y": {"$max": "$post_y"},
                        "min_post_x": {"$min": "$post_x"},
                        "max_post_x": {"$max": "$post_x"},
                    }
                }
            ]
            
            result = list(self._collection.aggregate(pipeline))
            if not result:
                # Empty database
                logger.warning("No synapses found in database")
                return Roi(Coordinate([0, 0, 0]), Coordinate([1000, 1000, 1000]))
                
            stats = result[0]
            
            # Calculate overall bounds
            min_z = min(stats["min_pre_z"], stats["min_post_z"])
            max_z = max(stats["max_pre_z"], stats["max_post_z"])
            min_y = min(stats["min_pre_y"], stats["min_post_y"])
            max_y = max(stats["max_pre_y"], stats["max_post_y"])
            min_x = min(stats["min_pre_x"], stats["min_post_x"])
            max_x = max(stats["max_pre_x"], stats["max_post_x"])
            
            # Add padding
            padding = 1000  # nm
            begin = Coordinate([min_z - padding, min_y - padding, min_x - padding])
            shape = Coordinate([
                max_z - min_z + 2 * padding,
                max_y - min_y + 2 * padding,
                max_x - min_x + 2 * padding
            ])
            
            roi = Roi(begin, shape)
            logger.info(f"Estimated ROI from database: {roi}")
            return roi
            
        except Exception as e:
            logger.error(f"Failed to estimate ROI: {e}")
            # Fallback
            return Roi(Coordinate([0, 0, 0]), Coordinate([100000, 100000, 100000]))
    
    def provide(self, request):
        """Provide synapse data for requested region."""
        # Get requested ROI
        synapse_roi = request[self.synapses_key].roi
        
        # Build MongoDB query for spatial selection
        query = {
            "$or": [
                # Presynaptic sites in ROI
                {
                    "pre_z": {"$gte": synapse_roi.begin[0], "$lt": synapse_roi.end[0]},
                    "pre_y": {"$gte": synapse_roi.begin[1], "$lt": synapse_roi.end[1]},
                    "pre_x": {"$gte": synapse_roi.begin[2], "$lt": synapse_roi.end[2]},
                },
                # Postsynaptic sites in ROI
                {
                    "post_z": {"$gte": synapse_roi.begin[0], "$lt": synapse_roi.end[0]},
                    "post_y": {"$gte": synapse_roi.begin[1], "$lt": synapse_roi.end[1]},
                    "post_x": {"$gte": synapse_roi.begin[2], "$lt": synapse_roi.end[2]},
                }
            ]
        }
        
        # Query synapses
        synapses = list(self._collection.find(query))
        logger.debug(f"Retrieved {len(synapses)} synapses for ROI {synapse_roi}")
        
        # Create graph
        graph = Graph([], [], GraphSpec(roi=synapse_roi, directed=True))
        
        node_id = 0
        edge_id = 0
        
        for syn_doc in synapses:
            # Create synapse object
            synapse = Synapse(
                id=syn_doc["id"],
                location_pre=np.array([syn_doc["pre_z"], syn_doc["pre_y"], syn_doc["pre_x"]]),
                location_post=np.array([syn_doc["post_z"], syn_doc["post_y"], syn_doc["post_x"]]),
                score=syn_doc.get("score"),
                id_segm_pre=syn_doc.get("pre_seg_id"),
                id_segm_post=syn_doc.get("post_seg_id")
            )
            
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
        
        # Create batch
        from gunpowder import Batch
        batch = Batch()
        batch[self.synapses_key] = graph
        
        return batch


class ZarrMongoDBDataSource:
    """
    Combined data source that loads raw volumes from Zarr and synapses from MongoDB.
    
    This provides the complete original Synful functionality for large-scale training
    on electron microscopy data with sparse synapse annotations.
    """
    
    def __init__(
        self,
        zarr_path: Union[str, Path],
        zarr_dataset: str,
        mongodb_host: str,
        mongodb_name: str,
        mongodb_collection: str,
        raw_key: "ArrayKey",
        synapses_key: "GraphKey",
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        Initialize combined Zarr + MongoDB source.
        
        Args:
            zarr_path: Path to Zarr volume
            zarr_dataset: Dataset name within Zarr (e.g., "volumes/raw")
            mongodb_host: MongoDB host
            mongodb_name: MongoDB database name
            mongodb_collection: MongoDB collection name
            raw_key: ArrayKey for raw data
            synapses_key: GraphKey for synapses
            voxel_size: Voxel size in nm
        """
        self.zarr_path = str(zarr_path)
        self.zarr_dataset = zarr_dataset
        self.mongodb_host = mongodb_host
        self.mongodb_name = mongodb_name
        self.mongodb_collection = mongodb_collection
        self.raw_key = raw_key
        self.synapses_key = synapses_key
        self.voxel_size = Coordinate(voxel_size)
        
    def create_pipeline(self):
        """Create the complete data source pipeline."""
        from gunpowder import ZarrSource, ArraySpec, MergeProvider
        
        # Raw volume source
        raw_source = ZarrSource(
            self.zarr_path,
            {self.raw_key: self.zarr_dataset},
            {self.raw_key: ArraySpec(interpolatable=True, voxel_size=self.voxel_size)}
        )
        
        # MongoDB synapse source
        synapse_source = MongoDBSynapseSource(
            db_host=self.mongodb_host,
            db_name=self.mongodb_name,
            collection_name=self.mongodb_collection,
            synapses_key=self.synapses_key,
            voxel_size=self.voxel_size
        )
        
        # Combine sources
        return (raw_source, synapse_source) + MergeProvider()


# Utility functions for easy integration
def create_zarr_mongodb_pipeline(
    zarr_path: str,
    mongodb_config: Dict[str, str],
    batch_size: Tuple[int, int, int] = (32, 256, 256),
    voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
    multitask: bool = True
):
    """
    Create a complete training pipeline with Zarr volumes and MongoDB synapses.
    
    Args:
        zarr_path: Path to Zarr volume
        mongodb_config: Dict with keys: host, database, collection
        batch_size: Training batch size in voxels
        voxel_size: Voxel size in nm
        multitask: Whether to generate direction vectors
        
    Returns:
        Configured Gunpowder pipeline ready for training
    """
    from gunpowder import ArrayKey, GraphKey
    
    # Define keys
    raw_key = ArrayKey('RAW')
    synapses_key = GraphKey('SYNAPSES')
    mask_key = ArrayKey('MASK')
    if multitask:
        direction_key = ArrayKey('DIRECTION')
    
    # Create data source
    source = ZarrMongoDBDataSource(
        zarr_path=zarr_path,
        zarr_dataset="volumes/raw",
        mongodb_host=mongodb_config["host"],
        mongodb_name=mongodb_config["database"],
        mongodb_collection=mongodb_config["collection"],
        raw_key=raw_key,
        synapses_key=synapses_key,
        voxel_size=voxel_size
    )
    
    return source.create_pipeline()