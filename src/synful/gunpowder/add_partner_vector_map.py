"""
Modernized AddPartnerVectorMap node for Synful.

This is a critical Gunpowder node that creates offset vectors pointing from source points 
(postsynaptic sites) to target points (presynaptic sites). This is the core functionality
that enables synaptic partner detection in Synful.

Modernized for Python 3.12+ while maintaining exact functional compatibility.
"""

import logging
from typing import Optional, Union, List

import numpy as np
import numpy.typing as npt

import gunpowder as gp
from gunpowder import BatchFilter, ArrayKey, GraphKey as PointsKey
from gunpowder import Array, ArraySpec, Coordinate, GraphSpec as PointsSpec, Roi
from gunpowder.morphology import enlarge_binary_map
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class AddPartnerVectorMap(BatchFilter):
    """
    Create an array with offset vectors pointing to target points.

    This node is essential for synaptic partner detection training. It creates
    dense vector fields around source points (typically postsynaptic sites) that
    point to their paired target points (typically presynaptic sites).

    Args:
        src_points: Source points around which vectors are created
        trg_points: Target points to which created vectors point to  
        array: The key of the array to create
        radius: Radius of the ball around source points in which vectors are created
        trg_context: Padding of trg_points request in world units to create src vectors
                    that point to target locations outside src roi
        mask: Optional array to mask the rasterization of source points. Contains
              discrete labels to intersect rasterization with specific objects
        pointmask: Optional key for a pointmask array marking blob regions where
                  vectors have been calculated
        array_spec: Optional spec of the array to create (datatype, voxel size)
    """

    def __init__(
        self,
        src_points: PointsKey,
        trg_points: PointsKey,
        array: ArrayKey,
        radius: Union[float, List[float], npt.NDArray],
        trg_context: Union[int, List[int], npt.NDArray],
        mask: Optional[ArrayKey] = None,
        pointmask: Optional[ArrayKey] = None,
        array_spec: Optional[ArraySpec] = None
    ):
        self.src_points = src_points
        self.trg_points = trg_points
        self.array = array
        self.radius = np.array([radius]).flatten().astype(np.float32)
        self.trg_context = np.array([trg_context]).flatten().astype(np.int32)
        self.mask = mask
        self.pointmask = pointmask
        
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec

    def setup(self):
        """Setup the node by configuring array specs and enabling autoskip."""
        src_roi = self.spec[self.src_points].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,) * src_roi.dims())

        if self.array_spec.dtype is None:
            self.array_spec.dtype = np.float32

        self.array_spec.roi = src_roi.copy()
        self.provides(self.array, self.array_spec)
        
        if self.pointmask is not None:
            self.provides(self.pointmask, self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):
        """Prepare the request by adding context for source and target points."""
        # For src point, use radius to determine the context
        context = np.ceil(self.radius).astype(np.int32)
        dims = self.array_spec.roi.dims()
        
        if len(context) == 1:
            context = context.repeat(dims)

        # Request points in a larger area
        src_roi = request[self.array].roi.grow(
            Coordinate(context),
            Coordinate(context)
        )

        # Restrict the request to the points actually provided
        src_roi_prov = src_roi.intersect(self.spec[self.src_points].roi)
        request[self.src_points] = PointsSpec(roi=src_roi_prov)

        # For trg points, use custom context option
        context = self.trg_context
        if len(context) == 1:
            context = context.repeat(dims)

        # Request points in a larger area
        trg_roi = src_roi.grow(
            Coordinate(context),
            Coordinate(context)
        )

        # Restrict the request to the points actually provided
        trg_roi = trg_roi.intersect(self.spec[self.trg_points].roi)
        request[self.trg_points] = PointsSpec(roi=trg_roi)

        # Handle mask if provided
        if self.mask is not None:
            mask_voxel_size = self.spec[self.mask].voxel_size
            assert self.spec[self.array].voxel_size == mask_voxel_size, (
                "Voxel size of mask and rasterized volume need to be equal"
            )

            new_mask_roi = src_roi.snap_to_grid(mask_voxel_size)
            # Restrict request to array provided
            new_mask_roi = new_mask_roi.intersect(self.spec[self.mask].roi)
            
            if self.mask in request:
                request[self.mask].roi = request[self.mask].roi.union(new_mask_roi)
            else:
                request[self.mask] = ArraySpec(roi=new_mask_roi)

    def process(self, batch, request):
        """Process the batch to create partner vector maps."""
        src_points = batch.points[self.src_points]
        voxel_size = self.spec[self.array].voxel_size

        # Get ROI used for creating the new array
        enlarged_vol_roi = src_points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin() / voxel_size
        shape = enlarged_vol_roi.get_shape() / voxel_size
        data_roi = Roi(offset, shape)

        logger.debug("Src points in %s", src_points.spec.roi)
        for i, point in src_points.data.items():
            logger.debug("%d, %s", i, point.location)
        logger.debug("Data roi in voxels: %s", data_roi)
        logger.debug("Data roi in world units: %s", data_roi * voxel_size)

        mask_array = None if self.mask is None else batch.arrays[self.mask].crop(enlarged_vol_roi)

        partner_vectors_data, pointmask = self._draw_partner_vectors(
            src_points,
            batch.points[self.trg_points],
            data_roi,
            voxel_size,
            enlarged_vol_roi.get_begin(),
            self.radius,
            mask_array
        )

        # Create array and crop it to requested roi
        spec = self.spec[self.array].copy()
        spec.roi = data_roi * voxel_size
        partner_vectors = Array(data=partner_vectors_data, spec=spec)
        
        logger.debug("Cropping partner vectors to %s", request[self.array].roi)
        batch.arrays[self.array] = partner_vectors.crop(request[self.array].roi)

        # Handle pointmask if requested
        if self.pointmask is not None and self.pointmask in request:
            spec = self.spec[self.array].copy()
            spec.roi = data_roi * voxel_size
            pointmask_array = Array(
                data=np.array(pointmask, dtype=spec.dtype),
                spec=spec
            )
            batch.arrays[self.pointmask] = pointmask_array.crop(request[self.pointmask].roi)

        # Restore requested ROI of src and target points
        if self.src_points in request:
            self._restore_points_roi(request, self.src_points, batch.points[self.src_points])
        if self.trg_points in request:
            self._restore_points_roi(request, self.trg_points, batch.points[self.trg_points])
            
        # Restore requested mask
        if self.mask is not None:
            batch.arrays[self.mask] = batch.arrays[self.mask].crop(request[self.mask].roi)

    def _restore_points_roi(self, request, points_key: PointsKey, points):
        """Restore the original ROI for points after processing."""
        request_roi = request[points_key].roi
        points.spec.roi = request_roi
        points.data = {
            i: p for i, p in points.data.items() 
            if request_roi.contains(p.location)
        }

    def _draw_partner_vectors(
        self, 
        src_points, 
        trg_points, 
        data_roi: Roi,
        voxel_size: Coordinate, 
        offset: Coordinate, 
        radius: npt.NDArray, 
        mask: Optional[Array] = None
    ):
        """
        Draw the actual partner vectors from sources to targets.
        
        This is the core algorithm that creates dense vector fields.
        """
        # 3D: z, y, x
        shape = data_roi.get_shape()
        logger.debug('Data roi %s', data_roi)
        d, h, w = shape

        # Create coordinate grids - 4D: c, z, y, x (c=[0, 1, 2])
        coords = np.array(
            np.meshgrid(
                np.arange(0, d),
                np.arange(0, h), 
                np.arange(0, w),
                indexing='ij'
            ),
            dtype=np.float32
        )

        # Convert to world coordinates
        coords[0, :] *= voxel_size[0]
        coords[1, :] *= voxel_size[1]
        coords[2, :] *= voxel_size[2]
        coords[0, :] += offset[0]
        coords[1, :] += offset[1]
        coords[2, :] += offset[2]

        target_vectors = np.zeros_like(coords)

        logger.debug("Adding vectors for %d points...", len(src_points.data))

        # For each src point, get a point mask
        union_mask = np.zeros(shape, dtype=np.int32)
        point_masks = []
        points_p = []
        targets = []
        
        for point_id, point in src_points.data.items():
            # Get the voxel coordinate
            v = Coordinate(point.location / voxel_size)

            if not data_roi.contains(v):
                logger.debug("Skipping point at %s outside of requested data ROI", v)
                continue

            assert len(point.partner_ids) == 1, (
                'AddPartnerVectorMap only implemented for single target point per src point'
            )
            trg_id = point.partner_ids[0]

            if trg_id not in trg_points.data:
                logger.warning("Target %d of %d not in trg points", trg_id, point_id)
                continue

            target = trg_points.data[trg_id]
            if not trg_points.spec.roi.contains(target.location):
                logger.warning(
                    "Target %d of %d not in target roi: %s", 
                    trg_id, point_id, trg_points.spec.roi
                )
                continue

            # Get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            if mask is not None:
                label = mask.data[v]
                object_mask = mask.data == label
                
            logger.debug(
                "Rasterizing point %s at %s",
                point.location,
                point.location / voxel_size - data_roi.get_begin()
            )

            point_mask = np.zeros(shape, dtype=bool)
            point_mask[v] = 1

            enlarge_binary_map(point_mask, radius, voxel_size, in_place=True)

            if mask is not None:
                point_mask &= object_mask
                
            union_mask += np.array(point_mask, dtype=np.int32)
            point_masks.append(point_mask)
            targets.append(target)
            points_p.append(v * voxel_size)

        assert len(targets) == len(points_p) == len(point_masks)
        
        if len(points_p) == 0:
            return target_vectors, np.array(union_mask, dtype=bool)

        # Remove overlap regions
        for point_mask in point_masks:
            point_mask[union_mask > 1] = False
            
        intersect_points = np.where(union_mask > 1)
        logger.debug('#voxels of overlapping src blobs: %d', len(intersect_points[0]))

        # Assign overlapping voxels to their closest src node
        if len(points_p) > 0:
            kd = KDTree(points_p)
            for intersect_point in zip(*intersect_points):
                p = Coordinate(intersect_point) * voxel_size
                dist, node_id = kd.query(p)
                point_masks[node_id][tuple(p // voxel_size)] = True

        # Calculate actual vectors with src blobs corrected for overlaps
        for ii, point_mask in enumerate(point_masks):
            target = targets[ii]
            target_vectors[0][point_mask] = target.location[0] - coords[0][point_mask]
            target_vectors[1][point_mask] = target.location[1] - coords[1][point_mask]
            target_vectors[2][point_mask] = target.location[2] - coords[2][point_mask]
            
        return target_vectors, np.array(union_mask, dtype=bool)