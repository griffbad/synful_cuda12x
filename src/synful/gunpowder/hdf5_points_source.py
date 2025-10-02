"""
Modernized Hdf5PointsSource for Synful pipeline.

This node provides HDF5-based point loading for pre- and post-synaptic markers
with full compatibility to the original implementation.
"""

import copy
import logging
from typing import Dict, Optional, Union

import numpy as np
import h5py

import gunpowder as gp
from gunpowder import Batch, BatchProvider, Coordinate, Graph as Points, GraphKey as PointsKey, GraphSpec as PointsSpec, Node
# from gunpowder.contrib.points import PreSynPoint, PostSynPoint  # May not exist in current version
from gunpowder.profiling import Timing

# Simple replacements for PreSynPoint and PostSynPoint
class PreSynPoint(Node):
    def __init__(self, location, location_id, synapse_id, partner_ids):
        super().__init__(id=location_id, location=location)
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids

class PostSynPoint(Node):
    def __init__(self, location, location_id, synapse_id, partner_ids):
        super().__init__(id=location_id, location=location)
        self.synapse_id = synapse_id
        self.partner_ids = partner_ids

logger = logging.getLogger(__name__)


class Hdf5PointsSource(BatchProvider):
    """
    An HDF5 data source for Points. Currently only supports a specific case 
    where points represent pre- and post-synaptic markers.

    Args:
        filename (str): The HDF5 file path.
        
        datasets (Dict[PointsKey, str]): Dictionary of PointsKey -> dataset names
            that this source offers.
            
        rois (Dict[PointsKey, gp.Roi], optional): Dictionary of PointsKey -> Roi 
            to set the ROI for each point set provided by this source.
            
        kind (str): Allowed arguments are 'synapse', 'presyn', 'postsyn'. 
            - 'synapse': provide two pointskeys: PRESYN and POSTSYN
            - 'presyn' or 'postsyn': only provide one pointkey
    """

    def __init__(
            self,
            filename: str,
            datasets: Dict[PointsKey, str],
            rois: Optional[Dict[PointsKey, gp.Roi]] = None,
            kind: str = 'synapse'):

        self.filename = filename
        self.datasets = datasets
        self.rois = rois
        self.kind = kind
        self.ndims = None
        
        assert kind in ['synapse', 'presyn', 'postsyn'], \
            f"option -kind- set to {kind}, Hdf5PointsSource implemented only " \
            f"for synapse, presyn or postsyn"
            
        if kind == 'synapse':
            assert len(datasets) == 2, \
                "option kind set to synapse, provide PointsKeys for Pre- and Postsynapse"
        else:
            assert len(datasets) == 1, \
                f"option kind set to {kind}, provide only one PointsKey"

    def setup(self):
        """Setup the provider by checking HDF5 file and registering point specs."""
        with h5py.File(self.filename, 'r') as hdf_file:
            for (points_key, ds_name) in self.datasets.items():
                if ds_name not in hdf_file:
                    raise RuntimeError(f"{ds_name} not in {self.filename}")

                spec = PointsSpec()
                if self.rois is not None and points_key in self.rois:
                    spec.roi = self.rois[points_key]

                self.provides(points_key, spec)

    def provide(self, request: gp.BatchRequest) -> Batch:
        """Provide points according to the request."""
        timing_process = Timing(self)
        timing_process.start()

        batch = Batch()

        with h5py.File(self.filename, 'r') as hdf_file:
            # Handle different point loading modes
            if any(key.name in ['PRESYN', 'POSTSYN'] for key in request.points_specs.keys()):
                # Legacy support for hardcoded PRESYN/POSTSYN keys
                points = self._get_synaptic_points_legacy(request, hdf_file)
            else:
                # Modern flexible point loading
                points = self._get_synaptic_points_modern(request, hdf_file)

            # Create Points objects for each requested key
            for (points_key, request_spec) in request.points_specs.items():
                logger.debug(f"Reading {points_key} in {request_spec.roi}...")
                points_spec = self.spec[points_key].copy()
                points_spec.roi = request_spec.roi
                
                if points_key in points:
                    logger.debug(f"Number of points: {len(points[points_key])}")
                    batch.points[points_key] = Points(
                        data=points[points_key], 
                        spec=points_spec
                    )

        timing_process.stop()
        batch.profiling_stats.add(timing_process)
        return batch

    def _get_synaptic_points_legacy(self, request: gp.BatchRequest, syn_file: h5py.File) -> Dict:
        """Legacy method for hardcoded PRESYN/POSTSYN keys."""
        # Find which keys are requested
        presyn_key = None
        postsyn_key = None
        
        for key in request.points_specs.keys():
            if key.name == 'PRESYN':
                presyn_key = key
            elif key.name == 'POSTSYN':
                postsyn_key = key
        
        # Use first requested key's ROI for both if only one is requested
        if presyn_key and not postsyn_key:
            pre_roi = post_roi = request.points_specs[presyn_key].roi
        elif postsyn_key and not presyn_key:
            pre_roi = post_roi = request.points_specs[postsyn_key].roi
        else:
            pre_roi = request.points_specs[presyn_key].roi
            post_roi = request.points_specs[postsyn_key].roi

        presyn_points, postsyn_points = self._extract_synaptic_points(
            pre_roi, post_roi, syn_file)
        
        points = {}
        if presyn_key:
            points[presyn_key] = presyn_points
        if postsyn_key:
            points[postsyn_key] = postsyn_points
            
        return points

    def _get_synaptic_points_modern(self, request: gp.BatchRequest, syn_file: h5py.File) -> Dict:
        """Modern method for flexible point key handling."""
        points = {}
        
        if self.kind == 'synapse':
            # For synapse kind, assume first key is presyn, second is postsyn
            keys = list(request.points_specs.keys())
            pre_key, post_key = keys[0], keys[1] if len(keys) > 1 else keys[0]
            
            pre_roi = request.points_specs[pre_key].roi
            post_roi = request.points_specs[post_key].roi if len(keys) > 1 else pre_roi
            
            presyn_points, postsyn_points = self._extract_synaptic_points(
                pre_roi, post_roi, syn_file)
            
            points[pre_key] = presyn_points
            if len(keys) > 1:
                points[post_key] = postsyn_points
                
        else:
            # For presyn/postsyn kind, load only the specified type
            synkey = list(self.datasets.keys())[0]
            roi = request.points_specs[synkey].roi
            
            presyn_points, postsyn_points = self._extract_synaptic_points(
                roi, roi, syn_file)
            
            points[synkey] = presyn_points if self.kind == 'presyn' else postsyn_points
            
        return points

    def _extract_synaptic_points(self, pre_roi: gp.Roi, post_roi: gp.Roi, syn_file: h5py.File):
        """Extract synaptic points from HDF5 file within specified ROIs."""
        presyn_points_dict = {}
        postsyn_points_dict = {}
        
        # Read annotations from HDF5
        annotation_ids = syn_file['annotations/ids'][:]
        locs = syn_file['annotations/locations'][:]
        
        # Handle optional offset
        offset = None
        if 'offset' in syn_file['annotations'].attrs:
            offset = np.array(syn_file['annotations'].attrs['offset'])
            logger.debug("Retrieving offset")
        else:
            logger.debug('No offset')

        # Process synaptic pairs
        syn_id = 0
        for pre, post in syn_file['annotations/presynaptic_site/partners'][:]:
            pre_index = int(np.where(pre == annotation_ids)[0][0])
            post_index = int(np.where(post == annotation_ids)[0][0])
            
            pre_site = locs[pre_index].copy()
            post_site = locs[post_index].copy()
            
            if offset is not None:
                pre_site += offset
                post_site += offset

            # Check if points are within requested ROIs
            pre_in_roi = pre_roi.contains(Coordinate(pre_site))
            post_in_roi = post_roi.contains(Coordinate(post_site))
            
            if pre_in_roi:
                syn_point = PreSynPoint(
                    location=pre_site,
                    location_id=pre_index,
                    synapse_id=syn_id,
                    partner_ids=[post_index]
                )
                presyn_points_dict[pre_index] = copy.deepcopy(syn_point)
                
            if post_in_roi:
                syn_point = PostSynPoint(
                    location=post_site,
                    location_id=post_index,
                    synapse_id=syn_id,
                    partner_ids=[pre_index]
                )
                postsyn_points_dict[post_index] = copy.deepcopy(syn_point)
                
            if pre_in_roi or post_in_roi:
                syn_id += 1

        return presyn_points_dict, postsyn_points_dict

    def __repr__(self):
        return f"Hdf5PointsSource({self.filename})"