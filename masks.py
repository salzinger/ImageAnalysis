import xarray as xr
from abc import ABCMeta, abstractmethod
import numpy as np

class Mask(metaclass=ABCMeta):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def get_mask(self, *args, **kwargs):
        pass

    def apply_mask(self, *args, **kwargs):
        return self.image.where(self.get_mask(*args, **kwargs))


@xr.register_dataset_accessor('rectangular_mask')
@xr.register_dataarray_accessor('rectangular_mask')
class RectangularMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=50, width_y=50):
        image = self.image
        mask = (abs(image.x - center_x) < width_x) * (abs(image.y - center_y) < width_y)
        return mask


@xr.register_dataset_accessor('polygon_mask')
@xr.register_dataarray_accessor('polygon_mask')
class PolygonMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, coord_vertices):
        try:
            image = self.image.isel(shot=0)
        except ValueError:
            image = self.image
        coord_vertices = np.array(coord_vertices)
        pos_indexers = [
            xr.core.coordinates.remap_label_indexers(image, dict(x=vertex[0], y=vertex[1]), method='nearest')[0] for
            vertex in coord_vertices]
        vertices = np.array([list(indexer.values()) for indexer in pos_indexers])
        shape = (len(image.x), len(image.y))

        polygon = _create_polygon(shape, vertices)
        mask = xr.DataArray(polygon, coords=image.coords)
        return mask


@xr.register_dataset_accessor('elliptical_mask')
@xr.register_dataarray_accessor('elliptical_mask')
class EllipticalMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=50, width_y=50, x='x', y='y', invert=False):
        image = self.image
        mask = (((image[x] - center_x)/width_x)**2 + ((image[y] - center_y)/width_y)**2) < 1
        if invert:
            return ~mask
        else:
            return mask


@xr.register_dataset_accessor('eit_mask')
@xr.register_dataarray_accessor('eit_mask')
class EITMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=200, width_y=600, width_eit=50):
        image = self.image
        mask_cloud = (((image.x - center_x)/width_x)**2 + ((image.y - center_y)/width_y)**2) < 1
        mask_spot = (((image.x - center_x)/width_eit)**2 + ((image.y - center_y)/width_eit)**2) > 1
        return mask_cloud*mask_spot


def _check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def _create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, _check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array