import numpy as np
from ..default_parameters import default_im_size


class ImageCrop():
    """Image crop, could be directly applied to numpy array slicing"""
    def __init__(self, 
                 ndim, 
                 crop_array=None,
                 single_im_size=default_im_size,
                 ):
        _shape = (ndim, 2)
        self.ndim = ndim
        self.array = np.zeros(_shape, dtype=np.int32)
        if crop_array is None:
            self.array[:,1] = np.array(single_im_size)
        else:
            self.update(crop_array)
        if len(single_im_size) == ndim:
            self.image_sizes = np.array(single_im_size, dtype=np.int32)
        
    def update(self, 
               crop_array, 
               ):
        _arr = np.array(crop_array, dtype=np.int32)
        if np.shape(_arr) == np.shape(self.array):
            self.array = _arr
        return
    
    def to_slices(self):
        return tuple([slice(_s[0], _s[1]) for _s in self.array])

    def inside(self, coords):
        """Check whether given coordinate is in this crop"""
        _coords = np.array(coords)
        if len(np.shape(_coords)) == 1:
            _coords = _coords[np.newaxis,:]
        elif len(np.shape(_coords)) > 2:
            raise IndexError("Only support single or multiple coordinates")
        # find kept spots
        _masks = [(_coords[:,_d] >= self.array[_d,0]) *\
                  (_coords[:,_d] <= self.array[_d,1])
                  for _d in range(self.ndim)]
        _mask = np.prod(_masks, axis=0).astype(bool)

        return _mask

    def distance_to_edge(self, coord):
        """Check distance of a coordinate to the edge of this crop"""
        _coord = np.array(coord)[:self.ndim]
        return np.min(np.abs(_coord[:,np.newaxis] - self.array))


    def crop_coords(self, coords):
        """ """
        _coords = np.array(coords)
        _mask = self.inside(coords)
        _cropped_coords = _coords[_mask] - self.array[:,0][np.newaxis,:]
        
        return _cropped_coords

    def overlap(self, crop2):
        
        # find overlaps
        _llim = np.max([self.array[:,0], crop2.array[:,0]], axis=0)
        
        _rlim = np.min([self.array[:,1], crop2.array[:,1]], axis=0)

        if (_llim > _rlim).any():
            return None
        else:
            return ImageCrop(len(_llim), np.array([_llim, _rlim]).transpose())

    def relative_overlap(self, crop2):
        _overlap = self.overlap(crop2)
        if _overlap is not None:
            _overlap.array = _overlap.array - self.array[:,0][:, np.newaxis]

        return _overlap

class ImageCrop_3d(ImageCrop):
    """3D image crop, could be directly applied to numpy array indexing """
    def __init__(self, 
                 crop_array=None,
                 single_im_size=default_im_size,
                 ):
    
        super().__init__(3, crop_array, single_im_size)

    def crop_spots(self, spots_3d):
        """ """
        _spots = spots_3d.copy()
        _coords = _spots[:,1:4]
        _mask = self.inside(_coords)
        _cropped_spots = _spots[_mask].copy()
        _cropped_spots[:,1:4] = np.array(_cropped_spots[:,1:4]) - self.array[:,0][np.newaxis,:]
        
        return _cropped_spots

    def overlap(self, crop2):
        _returned_crop = super().overlap(crop2)
        if _returned_crop is None:
            return None
        else:
            return ImageCrop_3d(_returned_crop.array)

    def translate_drift(self, drift=None):
        if drift is None:
            _drift = np.zeros(self.ndim, dtype=np.int32)
        else:
            _drift = np.round(drift).astype(np.int32)
        _new_box = []
        for _limits, _d, _sz in zip(self.array, _drift, self.image_sizes):
            _new_limits = [
                max(0, _limits[0]-_d),
                min(_sz, _limits[1]-_d),
            ]
            _new_box.append(np.array(_new_limits, dtype=np.int32))
        _new_box = np.array(_new_box)
        # generate new crop
        _new_crop = ImageCrop_3d(_new_box, self.image_sizes)
        #print(_drift, _new_box)
        return _new_crop

def generate_neighboring_crop(coord, crop_size=5, 
                              single_im_size=default_im_size,
                              sub_pixel_precision=False):
    """Function to generate crop given coord coordinate and crop size
    Inputs:
    Output:
    """
    ## check inputs
    _coord =  np.array(coord)[:len(single_im_size)]
    if isinstance(crop_size, int) or isinstance(crop_size, np.int32):
        _crop_size = np.ones(len(single_im_size),dtype=np.int32) * crop_size
    else:
        _crop_size = np.array(crop_size)[:len(single_im_size)]
        
    _single_image_size = np.array(single_im_size, dtype=np.int32)
    # find limits for this crop
    if sub_pixel_precision:
        _left_lims = np.max([_coord-_crop_size, np.zeros(len(_single_image_size))], axis=0)
        _right_lims = np.min([_coord+_crop_size+1, _single_image_size], axis=0)
    else:
        # limits
        _left_lims = np.max([np.round(_coord-_crop_size), np.zeros(len(_single_image_size))], axis=0)
        _right_lims = np.min([np.round(_coord+_crop_size+1), _single_image_size], axis=0)

    _crop = ImageCrop(len(single_im_size), 
                      np.array([_left_lims, _right_lims]).transpose(), 
                      single_im_size=single_im_size)

    return _crop

def translate_crop_by_drift(crop3d, drift3d=np.array([0,0,0]), single_im_size=default_im_size):
    """Function to translate 3d-crop by 3d-drift"""
    crop3d = np.array(crop3d, dtype=np.int)
    drift3d = np.array(drift3d)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # deal with negative upper limits    
    for _i, (_lims, _s) in enumerate(zip(crop3d, single_im_size)):
        if _lims[1] < 0:
            crop3d[_i,1] += _s
    _drift_limits = np.zeros(crop3d.shape, dtype=np.int)
    # generate drifted crops
    for _i, (_d, _lims) in enumerate(zip(drift3d, crop3d)):
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(_lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])
    return _drift_limits

def crop_neighboring_area(im, center, crop_sizes, 
                          extrapolate_mode='nearest'):
    
    """Function to crop neighboring area of a certain coordiante
    Args:
        im: image, np.ndarray
        center: zxy coordinate for the center of this crop area
        crop_sizes: dimension(s) of the cropping area, int or np.ndarray
        extrapolate_mode: mode in map_coordinate, str
    Return:
        _cim: cropped image, np.ndarray
    """
    if 'map_coordinates' not in locals():
        from scipy.ndimage import map_coordinates
    if not isinstance(im, np.ndarray):
        raise TypeError(f"wrong input image, should be np.ndarray")
    
    _dim = len(np.shape(im))
    _center = np.array(center)[:_dim]
    # crop size
    if isinstance(crop_sizes, int) or isinstance(crop_sizes, np.int32):
        _crop_sizes = np.ones(_dim, dtype=np.int32)*int(crop_sizes)
    elif isinstance(crop_sizes, list) or isinstance(crop_sizes, np.ndarray):
        _crop_sizes = np.array(crop_sizes)[:_dim]
    
    # generate a rough crop, to save RAM
    _rough_left_lims = np.max([np.zeros(_dim), 
                               np.floor(_center-_crop_sizes/2)], axis=0)
    _rough_right_lims = np.min([np.array(np.shape(im)), 
                                np.ceil(_center+_crop_sizes/2)], axis=0)
    _rough_center = _center - _rough_left_lims
    
    _rough_crop = tuple([slice(int(_l),int(_r)) for _l,_r in zip(_rough_left_lims, _rough_right_lims)])
    _rough_cropped_im = im[_rough_crop]
    
    # generate coordinates to be mapped
    _pixel_coords = np.indices(_crop_sizes) + np.expand_dims(_rough_center - (_crop_sizes-1)/2, 
                                                       tuple(np.arange(_dim)+1))
    #return _pixel_coords
    # map coordiates
    _cim = map_coordinates(_rough_cropped_im, _pixel_coords.reshape(_dim, -1),
                           mode=extrapolate_mode, cval=np.min(_rough_cropped_im))
    _cim = _cim.reshape(_crop_sizes) # reshape back to original shape
    
    return _cim


