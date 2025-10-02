"""
TSV data source for Synful.

Provides support for simple TSV files containing synapse locations,
where each row represents one synapse with pre and post coordinates.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from gunpowder import BatchProvider, GraphSpec, GraphKey, Graph, Node, Edge, Roi, Coordinate

# Conditional import to avoid circular dependency
try:
    from ..synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError:
    _SYNAPSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TSVSynapseSource(BatchProvider):
    """
    TSV source for synapse annotations.
    
    Expects a TSV file with columns:
    pre_x, pre_y, pre_z, post_x, post_y, post_z
    
    Each row represents one synapse with pre and post locations.
    """
    
    def __init__(
        self,
        tsv_path: Union[str, Path],
        synapses_key: GraphKey,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        coordinate_order: str = "xyz",  # or "zyx" for z,y,x order
        has_header: bool = True,
        roi: Optional[Roi] = None
    ):
        """
        Initialize TSV synapse source.
        
        Args:
            tsv_path: Path to TSV file
            synapses_key: GraphKey for synapse data
            voxel_size: Voxel size in nm (z, y, x)
            coordinate_order: Order of coordinates in file ("xyz" or "zyx")
            has_header: Whether TSV has header row
            roi: Optional ROI to restrict synapses to
        """
        self.tsv_path = Path(tsv_path)
        self.synapses_key = synapses_key
        self.voxel_size = Coordinate(voxel_size)
        self.coordinate_order = coordinate_order.lower()
        self.has_header = has_header
        self.roi = roi
        
        # Load synapse data
        self._synapses = []
        self._load_synapses()
        
    def _load_synapses(self):
        """Load synapses from TSV file."""
        try:
            # Read TSV file
            if self.has_header:
                df = pd.read_csv(self.tsv_path, sep='\t')
            else:
                # Assume column order based on coordinate_order
                if self.coordinate_order == "xyz":
                    columns = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']
                else:  # zyx
                    columns = ['pre_z', 'pre_y', 'pre_x', 'post_z', 'post_y', 'post_x']
                df = pd.read_csv(self.tsv_path, sep='\t', header=None, names=columns)
            
            logger.info(f"Loaded TSV with {len(df)} synapses from {self.tsv_path}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Convert to synapse objects
            for idx, row in df.iterrows():
                # Extract coordinates based on order
                if self.coordinate_order == "xyz":
                    # Convert x,y,z to z,y,x for internal use
                    pre_loc = np.array([row['pre_z'], row['pre_y'], row['pre_x']], dtype=float)
                    post_loc = np.array([row['post_z'], row['post_y'], row['post_x']], dtype=float)
                else:  # zyx
                    pre_loc = np.array([row['pre_z'], row['pre_y'], row['pre_x']], dtype=float)
                    post_loc = np.array([row['post_z'], row['post_y'], row['post_x']], dtype=float)
                
                synapse = Synapse(
                    id=int(idx),
                    location_pre=pre_loc,
                    location_post=post_loc,
                    score=row.get('score', 1.0)  # Default score if not provided
                )
                
                self._synapses.append(synapse)
                
            logger.info(f"Created {len(self._synapses)} synapse objects")
            
            # Log coordinate ranges
            if self._synapses:
                pre_coords = np.array([s.location_pre for s in self._synapses])
                post_coords = np.array([s.location_post for s in self._synapses])
                all_coords = np.vstack([pre_coords, post_coords])
                
                logger.info(f"Coordinate ranges (z,y,x):")
                logger.info(f"  Min: {all_coords.min(axis=0)}")
                logger.info(f"  Max: {all_coords.max(axis=0)}")
                logger.info(f"  Extent: {all_coords.max(axis=0) - all_coords.min(axis=0)}")
                
        except Exception as e:
            logger.error(f"Failed to load TSV file {self.tsv_path}: {e}")
            raise
            
    def setup(self):
        """Setup the provider."""
        # Calculate ROI from synapses if not provided
        if self.roi is None:
            self.roi = self._estimate_roi()
            
        self.provides(
            self.synapses_key,
            GraphSpec(roi=self.roi, directed=True)
        )
        
    def _estimate_roi(self) -> Roi:
        """Estimate ROI from loaded synapses."""
        if not self._synapses:
            logger.warning("No synapses loaded")
            return Roi(Coordinate([0, 0, 0]), Coordinate([1000, 1000, 1000]))
        
        # Get all coordinates
        all_coords = []
        for synapse in self._synapses:
            all_coords.extend([synapse.location_pre, synapse.location_post])
        
        all_coords = np.array(all_coords)
        min_coords = all_coords.min(axis=0)
        max_coords = all_coords.max(axis=0)
        
        # Add padding
        padding = 1000  # nm
        begin = Coordinate(min_coords - padding)
        shape = Coordinate(max_coords - min_coords + 2 * padding)
        
        roi = Roi(begin, shape)
        logger.info(f"Estimated ROI from TSV: {roi}")
        return roi
    
    def provide(self, request):
        """Provide synapse data for requested region."""
        # Get requested ROI
        synapse_roi = request[self.synapses_key].roi
        
        # Filter synapses that overlap with ROI
        filtered_synapses = []
        for synapse in self._synapses:
            pre_in_roi = synapse_roi.contains(Coordinate(synapse.location_pre))
            post_in_roi = synapse_roi.contains(Coordinate(synapse.location_post))
            
            if pre_in_roi or post_in_roi:
                filtered_synapses.append(synapse)
        
        logger.debug(f"Filtered to {len(filtered_synapses)} synapses for ROI {synapse_roi}")
        
        # Create graph
        graph = Graph([], [], GraphSpec(roi=synapse_roi, directed=True))
        
        node_id = 0
        edge_id = 0
        
        for synapse in filtered_synapses:
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


def load_synapses_from_tsv(
    tsv_path: Union[str, Path],
    coordinate_order: str = "xyz",
    has_header: bool = True
) -> SynapseCollection:
    """
    Load synapses from TSV file into a SynapseCollection.
    
    Args:
        tsv_path: Path to TSV file
        coordinate_order: Order of coordinates ("xyz" or "zyx")
        has_header: Whether file has header row
        
    Returns:
        SynapseCollection with loaded synapses
    """
    tsv_path = Path(tsv_path)
    
    # Read TSV
    if has_header:
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        if coordinate_order.lower() == "xyz":
            columns = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']
        else:
            columns = ['pre_z', 'pre_y', 'pre_x', 'post_z', 'post_y', 'post_x']
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=columns)
    
    # Convert to synapses
    synapses = []
    for idx, row in df.iterrows():
        if coordinate_order.lower() == "xyz":
            # Convert x,y,z to z,y,x
            pre_loc = np.array([row['pre_z'], row['pre_y'], row['pre_x']], dtype=float)
            post_loc = np.array([row['post_z'], row['post_y'], row['post_x']], dtype=float)
        else:
            pre_loc = np.array([row['pre_z'], row['pre_y'], row['pre_x']], dtype=float)
            post_loc = np.array([row['post_z'], row['post_y'], row['post_x']], dtype=float)
        
        synapse = Synapse(
            id=int(idx),
            location_pre=pre_loc,
            location_post=post_loc,
            score=row.get('score', 1.0)
        )
        synapses.append(synapse)
    
    return SynapseCollection(synapses)