import numpy as np
from scipy.spatial.distance import cdist, pdist
# default params
_3d_spot_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']
_3d_infos = ['z', 'x', 'y']
_spot_coord_inds = [_3d_spot_infos.index(_info) for _info in _3d_infos]

class Spots3D(np.ndarray):
    """Class for fitted spots in 3D"""
    def __new__(cls, 
                input_array, 
                bits=None,
                pixel_sizes=None,
                channels=None,
                copy_data=True,
                intensity_index=0,
                coordinate_indices=[1,2,3]):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if copy_data:
            input_array = np.array(input_array).copy()
        if len(np.shape(input_array)) == 1:
            obj = np.asarray([input_array]).view(cls)
        elif len(np.shape(input_array)) == 2:
            obj = np.asarray(input_array).view(cls)
        else:
            raise IndexError('Spots3D class only creating 2D-array')
        # add the new attribute to the created instance
        if isinstance(bits, (int, np.int32)):
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits)
        elif bits is not None and np.size(bits) == 1:
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits[0])
        elif bits is not None and len(bits) == len(obj):
            obj.bits = np.array(bits, dtype=np.int32) 
        else:
            obj.bits = bits
        # channels
        if isinstance(channels, bytes):
            channels = channels.decode()
        if isinstance(channels, (int, np.int32)):
            obj.channels = np.ones(len(obj), dtype=np.int32) * int(channels)
        elif channels is not None and isinstance(channels, str):
            obj.channels = np.array([channels]*len(obj))
        elif channels is not None and len(channels) == len(obj):
            obj.channels = np.array(channels) 
        else:
            obj.channels = channels
        # others
        obj.pixel_sizes = np.array(pixel_sizes)
        obj.intensity_index = int(intensity_index)
        obj.coordinate_indices = np.array(coordinate_indices, dtype=np.int32)
        # default parameters
        obj._3d_infos = _3d_infos
        obj._3d_spot_infos = _3d_spot_infos
        obj._spot_coord_inds = np.array(_spot_coord_inds)
        # Finally, we must return the newly created object:
        return obj

    #    def __str__(self):
    #        """Spots3D object with dimension"""
    #        return ""

    def __getitem__(self, key):
        """Modified getitem to allow slicing of bits as well"""
        #print(f" getitem {key}, {type(key)}", self.shape)
        new_obj = super().__getitem__(key)
        # if slice, slice bits as well
        if hasattr(self, 'bits') and getattr(self, 'bits') is not None and len(np.shape(getattr(self, 'bits')))==1:
            if isinstance(key, slice) or isinstance(key, np.ndarray) or isinstance(key, int):
                setattr(new_obj, 'bits', getattr(self, 'bits')[key] )
        if hasattr(self, 'channels') and getattr(self, 'channels') is not None and len(np.shape(getattr(self, 'channels')))==1:
            if isinstance(key, slice) or isinstance(key, np.ndarray) or isinstance(key, int):
                setattr(new_obj, 'channels', getattr(self, 'channels')[key] )
        #print(new_obj, type(new_obj))
        return new_obj

    def __setitem__(self, key, value):
        #print(f" setitem {key}, {type(key)}")
        return super().__setitem__(key, value)

    def __array_finalize__(self, obj):
        """
        Reference: https://numpy.org/devdocs/user/basics.subclassing.html 
        """
        if obj is None: 
            return
        else:
            if hasattr(obj, 'shape') and len(getattr(obj, 'shape')) != 2:
                obj = np.array(obj)
            # other attributes
            setattr(self, 'bits', getattr(obj, 'bits', None))
            setattr(self, 'channels', getattr(obj, 'channels', None))
            setattr(self, 'pixel_sizes', getattr(obj, 'pixel_sizes', None))
            setattr(self, 'intensity_index', getattr(obj, 'intensity_index', None))
            setattr(self, 'coordinate_indices', getattr(obj, 'coordinate_indices', None))
            setattr(self, '_3d_infos', getattr(obj, '_3d_infos', None))
            setattr(self, '_3d_spot_infos', getattr(obj, '_3d_spot_infos', None))
            setattr(self, '_spot_coord_inds', getattr(obj, '_spot_coord_inds', None))
        #print(f"**finalizing, {obj}, {type(obj)}")
        return obj

    def to_coords(self):
        """ convert into 3D coordinates in pixels """
        _coordinate_indices = getattr(self, 'coordinate_indices', np.array([1,2,3]))
        if len(np.shape(self)) > 1:
            return np.array(self[:,_coordinate_indices])
        else:
            return np.array(self[_coordinate_indices])
    
    def to_positions(self, pixel_sizes=None):
        """ convert into 3D spatial positions"""
        _saved_pixel_sizes = getattr(self, 'pixel_sizes', None)
        if _saved_pixel_sizes is not None and _saved_pixel_sizes.any():
            return self.to_coords() * np.array(_saved_pixel_sizes)[np.newaxis,:]
        elif pixel_sizes is None:
            raise ValueError('pixel_sizes not given')
        else:
            return self.to_coords() * np.array(pixel_sizes)[np.newaxis,:]

    def to_intensities(self):
        """ """
        _intensity_index = getattr(self, 'intensity_index', 0 )
        return np.array(self[:,_intensity_index])

# scoring spot Tuple
class SpotTuple():
    """Tuple of coordinates"""
    def __init__(self, 
                 spots_tuple:Spots3D,
                 bits:np.ndarray=None,
                 pixel_sizes:np.ndarray or list=None,
                 spots_inds=None,
                 tuple_id=None,
                 ):
        # add spot Tuple
        self.spots = spots_tuple[:].copy()
        # add information for bits
        if isinstance(bits, int):
            self.bits = np.ones(len(self.spots), dtype=np.int32) * int(bits)
        elif bits is not None and np.size(bits) == 1:
            self.bits = np.ones(len(self.spots), dtype=np.int32) * int(bits[0])
        elif bits is not None:
            self.bits = np.array(bits[:len(self.spots)], dtype=np.int32) 
        elif spots_tuple.bits is not None:
            self.bits = spots_tuple.bits[:len(self.spots)]
        else:
            self.bits = bits
        if pixel_sizes is None:
            self.pixel_sizes = getattr(self.spots, 'pixel_sizes', None)
        else:
            self.pixel_sizes = np.array(pixel_sizes)
        
        self.spots_inds = spots_inds
        self.tuple_id = tuple_id
        
    def dist_internal(self):
        _self_coords = self.spots.to_positions(self.pixel_sizes)
        return pdist(_self_coords)

    def intensities(self):
        return self.spots.to_intensities()
    def intensity_mean(self):
        return np.mean(self.spots.to_intensities())

    def centroid_spot(self):
        self.centroid = np.mean(self.spots, axis=0, keepdims=True)
        self.centroid.pixel_sizes = self.pixel_sizes
        return self.centroid

    def dist_centroid_to_spots(self, spots:Spots3D):
        """Calculate distance from tuple centroid to given spots"""
        if not hasattr(self, 'centroid'):
            _cp = self.centroid_spot()
        else:
            _cp = getattr(self, 'centroid')
        _centroid_coords = _cp.to_positions(pixel_sizes=self.pixel_sizes)
        _target_coords = spots.to_positions(pixel_sizes=self.pixel_sizes)
        return cdist(_centroid_coords, _target_coords)[0]

    def dist_to_spots(self, 
                      spots:Spots3D):
        _self_coords = self.spots.to_positions(pixel_sizes=self.pixel_sizes)
        _target_coords = spots.to_positions(pixel_sizes=self.pixel_sizes)
        return cdist(_self_coords, _target_coords)

    def dist_chromosome(self):
        pass
