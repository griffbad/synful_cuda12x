"""
Modern synapse data structures with Pydantic validation and enhanced functionality.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import torch
from pathlib import Path
import h5py
import zarr


class Synapse(BaseModel):
    """
    Modern synapse representation with enhanced validation and functionality.
    """
    
    id: Optional[Union[int, str]] = None
    id_segm_pre: Optional[int] = None
    id_segm_post: Optional[int] = None
    location_pre: Optional[Tuple[float, float, float]] = None
    location_post: Optional[Tuple[float, float, float]] = None
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    id_skel_pre: Optional[int] = None
    id_skel_post: Optional[int] = None
    node_id_pre: Optional[int] = None
    node_id_post: Optional[int] = None
    
    # Additional modern fields
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    direction_vector: Optional[Tuple[float, float, float]] = None
    distance: Optional[float] = Field(None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        
    @validator('location_pre', 'location_post', 'direction_vector')
    def validate_coordinates(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError('Coordinates must be 3D (x, y, z)')
        return v
        
    @property
    def has_complete_pair(self) -> bool:
        """Check if synapse has both pre and post synaptic information."""
        return (
            self.location_pre is not None and 
            self.location_post is not None and
            self.id_segm_pre is not None and 
            self.id_segm_post is not None
        )
        
    @property
    def euclidean_distance(self) -> Optional[float]:
        """Calculate Euclidean distance between pre and post synaptic sites."""
        if self.location_pre is None or self.location_post is None:
            return None
            
        return float(np.linalg.norm(
            np.array(self.location_pre) - np.array(self.location_post)
        ))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict(exclude_none=True)
        
    def __str__(self) -> str:
        """Enhanced string representation."""
        parts = []
        if self.id is not None:
            parts.append(f"id: {self.id}")
        if self.id_segm_pre is not None and self.id_segm_post is not None:
            parts.append(f"seg_ids: [{self.id_segm_pre}, {self.id_segm_post}]")
        if self.score is not None:
            parts.append(f"score: {self.score:.3f}")
        if self.confidence is not None:
            parts.append(f"confidence: {self.confidence:.3f}")
        if self.euclidean_distance is not None:
            parts.append(f"distance: {self.euclidean_distance:.1f}")
            
        return f"Synapse({', '.join(parts)})"


class SynapseCollection:
    """
    Collection of synapses with batch operations and analysis functionality.
    """
    
    def __init__(self, synapses: List[Synapse] = None):
        self.synapses = synapses or []
        
    def __len__(self) -> int:
        return len(self.synapses)
        
    def __getitem__(self, index: Union[int, slice]) -> Union[Synapse, 'SynapseCollection']:
        if isinstance(index, slice):
            return SynapseCollection(self.synapses[index])
        return self.synapses[index]
        
    def __iter__(self):
        return iter(self.synapses)
        
    def add(self, synapse: Synapse) -> None:
        """Add a synapse to the collection."""
        self.synapses.append(synapse)
        
    def extend(self, synapses: List[Synapse]) -> None:
        """Add multiple synapses to the collection."""
        self.synapses.extend(synapses)
        
    def filter_by_score(self, min_score: float = 0.0, max_score: float = 1.0) -> 'SynapseCollection':
        """Filter synapses by score range."""
        filtered = [
            s for s in self.synapses 
            if s.score is not None and min_score <= s.score <= max_score
        ]
        return SynapseCollection(filtered)
        
    def filter_by_confidence(self, min_confidence: float = 0.0) -> 'SynapseCollection':
        """Filter synapses by minimum confidence."""
        filtered = [
            s for s in self.synapses 
            if s.confidence is not None and s.confidence >= min_confidence
        ]
        return SynapseCollection(filtered)
        
    def filter_complete_pairs(self) -> 'SynapseCollection':
        """Filter to only complete synaptic pairs."""
        filtered = [s for s in self.synapses if s.has_complete_pair]
        return SynapseCollection(filtered)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collection to pandas DataFrame."""
        data = []
        for synapse in self.synapses:
            row = synapse.to_dict()
            # Flatten coordinate tuples
            if row.get('location_pre'):
                row['pre_x'], row['pre_y'], row['pre_z'] = row.pop('location_pre')
            if row.get('location_post'):
                row['post_x'], row['post_y'], row['post_z'] = row.pop('location_post')
            if row.get('direction_vector'):
                row['dir_x'], row['dir_y'], row['dir_z'] = row.pop('direction_vector')
            data.append(row)
            
        return pd.DataFrame(data)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary for JSON serialization."""
        return {
            'num_synapses': len(self.synapses),
            'synapses': [synapse.to_dict() for synapse in self.synapses],
            'metadata': {
                'created_at': str(pd.Timestamp.now()),
                'version': '1.0'
            }
        }
        
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays for efficient processing."""
        df = self.to_dataframe()
        return {col: df[col].values for col in df.columns}
        
    def to_torch(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        arrays = self.to_numpy()
        tensors = {}
        for key, array in arrays.items():
            if array.dtype in [np.float32, np.float64, np.int32, np.int64]:
                tensors[key] = torch.tensor(array, device=device)
        return tensors
        
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """Save collection to HDF5 file."""
        filepath = Path(filepath)
        df = self.to_dataframe()
        
        with h5py.File(filepath, 'w') as f:
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Handle string columns
                    data = df[column].astype(str).values
                    f.create_dataset(column, data=data)
                else:
                    f.create_dataset(column, data=df[column].values)
                    
    def save_zarr(self, filepath: Union[str, Path]) -> None:
        """Save collection to Zarr format."""
        filepath = Path(filepath)
        df = self.to_dataframe()
        
        store = zarr.DirectoryStore(str(filepath))
        root = zarr.group(store=store, overwrite=True)
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Handle string columns
                data = df[column].astype(str).values
                root.create_dataset(column, data=data, dtype='<U50')
            else:
                root.create_dataset(column, data=df[column].values)
                
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'SynapseCollection':
        """Load collection from HDF5 file."""
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][:]
                
        # Reconstruct coordinate tuples
        synapses = []
        n_synapses = len(data[list(data.keys())[0]])
        
        for i in range(n_synapses):
            synapse_data = {}
            for key, values in data.items():
                if key.endswith(('_x', '_y', '_z')):
                    continue
                synapse_data[key] = values[i] if values[i] is not None else None
                
            # Reconstruct coordinates
            if 'pre_x' in data and 'pre_y' in data and 'pre_z' in data:
                synapse_data['location_pre'] = (
                    float(data['pre_x'][i]), 
                    float(data['pre_y'][i]), 
                    float(data['pre_z'][i])
                )
            if 'post_x' in data and 'post_y' in data and 'post_z' in data:
                synapse_data['location_post'] = (
                    float(data['post_x'][i]), 
                    float(data['post_y'][i]), 
                    float(data['post_z'][i])
                )
            if 'dir_x' in data and 'dir_y' in data and 'dir_z' in data:
                synapse_data['direction_vector'] = (
                    float(data['dir_x'][i]), 
                    float(data['dir_y'][i]), 
                    float(data['dir_z'][i])
                )
                
            synapses.append(Synapse(**synapse_data))
            
        return cls(synapses)
        
    @classmethod
    def load_zarr(cls, filepath: Union[str, Path]) -> 'SynapseCollection':
        """Load collection from Zarr format."""
        filepath = Path(filepath)
        
        store = zarr.DirectoryStore(str(filepath))
        root = zarr.group(store=store)
        
        data = {}
        for key in root.keys():
            data[key] = root[key][:]
            
        # Similar reconstruction logic as HDF5
        synapses = []
        n_synapses = len(data[list(data.keys())[0]])
        
        for i in range(n_synapses):
            synapse_data = {}
            for key, values in data.items():
                if key.endswith(('_x', '_y', '_z')):
                    continue
                synapse_data[key] = values[i] if values[i] is not None else None
                
            # Reconstruct coordinates (same as HDF5)
            if 'pre_x' in data and 'pre_y' in data and 'pre_z' in data:
                synapse_data['location_pre'] = (
                    float(data['pre_x'][i]), 
                    float(data['pre_y'][i]), 
                    float(data['pre_z'][i])
                )
            if 'post_x' in data and 'post_y' in data and 'post_z' in data:
                synapse_data['location_post'] = (
                    float(data['post_x'][i]), 
                    float(data['post_y'][i]), 
                    float(data['post_z'][i])
                )
            if 'dir_x' in data and 'dir_y' in data and 'dir_z' in data:
                synapse_data['direction_vector'] = (
                    float(data['dir_x'][i]), 
                    float(data['dir_y'][i]), 
                    float(data['dir_z'][i])
                )
                
            synapses.append(Synapse(**synapse_data))
            
        return cls(synapses)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of the collection."""
        df = self.to_dataframe()
        
        stats = {
            'total_synapses': len(self),
            'complete_pairs': len(self.filter_complete_pairs()),
            'with_scores': len([s for s in self.synapses if s.score is not None]),
            'with_confidence': len([s for s in self.synapses if s.confidence is not None]),
        }
        
        # Score statistics
        scores = [s.score for s in self.synapses if s.score is not None]
        if scores:
            stats['score_stats'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
            }
            
        # Distance statistics
        distances = [s.euclidean_distance for s in self.synapses if s.euclidean_distance is not None]
        if distances:
            stats['distance_stats'] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'median': np.median(distances),
            }
            
        return stats