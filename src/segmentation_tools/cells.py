import os, cv2, h5py, copy, math
import numpy as np
from scipy.ndimage import grey_dilation

from ..file_io.merlin_params import _read_microscope_json

default_cellpose_kwargs = {
    'anisotropy': 1,
    'diameter': 60,
    'min_size': 200,
    'stitch_threshold': 0.1,
    'do_3D':True,
}
default_alignment_params = {
    'dialation_size':4,
}
default_pixel_sizes = [250,108,108]
default_Zcoords = np.arange(13)
default_dna_Zcoords = np.round(np.arange(0,12.5,0.25),2)

class Align_Segmentation():
    """
    Align segmentation from remounted RNA-DNA sample
    """
    def __init__(self, 
        rna_feature_file:str, 
        rna_dapi_file:str, 
        dna_save_file:str,
        rna_microscope_file:str,
        dna_microscope_file:str,
        rotation_mat:np.ndarray, #
        parameters:dict={},
        overwrite:bool=False,
        debug:bool=False,
        verbose:bool=True,
        ):
        self.rna_feature_file = rna_feature_file
        self.rna_dapi_file = rna_dapi_file
        self.dna_save_file = dna_save_file
        self.rna_microscope_file = rna_microscope_file
        self.dna_microscope_file = dna_microscope_file
        self.rotation_mat = rotation_mat
        # params
        self.parameters = {_k:_v for _k,_v in default_alignment_params.items()}
        self.parameters.update(parameters)
        self.overwrite = overwrite
        self.debug = debug
        self.verbose = verbose

    @staticmethod
    def _load_rna_feature(rna_feature_file:str, _z_coords=default_Zcoords):
        """Load RNA feature from my MERLIN output"""
        _fovcell_2_uid = {}
        with h5py.File(rna_feature_file, 'r') as _f:
            _label_group = _f['labeldata']
            rna_mask = _label_group['label3D'][:] # read
            #rna_mask = Align_Segmentation._correct_image3D_by_microscope_param(rna_mask, microscpe_params) # transpose and flip
            if np.max(rna_mask) <= 0:
                print(f'No cell found in feature file: {rna_feature_file}')
                return rna_mask, _fovcell_2_uid
            else:
                # load feature info
                _feature_group = _f['featuredata']
                for _cell_uid in _feature_group.keys():
                    _cell_group = _feature_group[_cell_uid]
                    _z_coords = _cell_group['z_coordinates'][:]
                    _fovcell_2_uid[(_cell_group.attrs['fov'], _cell_group.attrs['label'])] = _cell_uid
        return rna_mask, _z_coords, _fovcell_2_uid

    @staticmethod
    def _load_dna_info(dna_save_file:str, microscpe_params:dict):
        # Load DAPI
        with h5py.File(dna_save_file, "r", libver='latest') as _f:
            _fov_id = _f.attrs['fov_id']
            _fov_name = _f.attrs['fov_name']
            # load DAPI
            if 'dapi_im' in _f.attrs.keys():
                _dapi_im = _f.attrs['dapi_im']
                # translate DNA
                _dapi_im = Align_Segmentation._correct_image3D_by_microscope_param(_dapi_im, microscpe_params) # transpose and flip
            else:
                _dapi_im = None
        return _dapi_im, _fov_id, _fov_name
    @staticmethod
    def _load_rna_dapi(rna_dapi_file:str, microscpe_params:dict):
        _rna_dapi = np.load(rna_dapi_file)
        _rna_dapi = Align_Segmentation._correct_image3D_by_microscope_param(_rna_dapi, microscpe_params) # transpose and flip
        return _rna_dapi
    @staticmethod
    def _read_microscope_json(_microscope_file:str,):
        return _read_microscope_json(_microscope_file)

    @staticmethod
    def _correct_image3D_by_microscope_param(image3D:np.ndarray, microscope_params:dict):
        """Correct 3D image with microscopy parameter"""
        _image = copy.copy(image3D)
        if not isinstance(microscope_params, dict):
            raise TypeError(f"Wrong inputt ype for microscope_params, should be a dict")
        # transpose
        if 'transpose' in microscope_params and microscope_params['transpose']:
            _image = _image.transpose((0,2,1))
        if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
            _image = np.flip(_image, 2)
        if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
            _image = np.flip(_image, 1)
        return _image

    @staticmethod
    def _correct_image2D_by_microscope_param(image2D:np.ndarray, microscope_params:dict):
        """Correct 3D image with microscopy parameter"""
        _image = copy.copy(image2D)
        # transpose
        if 'transpose' in microscope_params and microscope_params['transpose']:
            _image = _image.transpose((1,0))
        if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
            _image = np.flip(_image, 1)
        if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
            _image = np.flip(_image, 0)
        return _image

    def _generate_dna_mask(self, target_dna_Zcoords=default_dna_Zcoords, save_dtype=np.uint16):
        # process microscope.json
        _rna_mparam = _read_microscope_json(self.rna_microscope_file)
        _dna_mparam = _read_microscope_json(self.dna_microscope_file)
        # load RNA
        _rna_mask, _rna_Zcoords, _fovcell_2_uid = self._load_rna_feature(self.rna_feature_file,)
        _rna_dapi = self._load_rna_dapi(self.rna_dapi_file, _rna_mparam)
        # generate full
        _full_rna_mask = interploate_z_masks(_rna_mask, _rna_Zcoords, 
                                             target_dna_Zcoords, verbose=self.verbose)
        # load DNA
        _dna_dapi, _fov_id, _fov_name = self._load_dna_info(self.dna_save_file, _dna_mparam)
        # decide rotation matrix
        #f _dna_mparam.get('transpose', True):
        #    _dna_rot_mat = self.rotation_mat.transpose()
            
        # translate
        _dna_mask, _rot_dna_dapi = translate_segmentation(
            _rna_dapi, _dna_dapi, self.rotation_mat, 
            label_before=_full_rna_mask, 
            return_new_dapi=True, verbose=self.verbose)
        # Do dialation
        if 'dialation_size' in self.parameters:
            _dna_mask = grey_dilation(_dna_mask, size=self.parameters['dialation_size'])
        _dna_mask = np.clip(_dna_mask, np.iinfo(save_dtype).min, np.iinfo(save_dtype).max,)
        _dna_mask = _dna_mask.astype(save_dtype)
        # add to attribute
        self.dna_mask = _dna_mask
        self.fov_id = _fov_id
        self.fov_name = _fov_name
        self.fovcell_2_uid = _fovcell_2_uid
        if self.debug:
            return _dna_mask, _full_rna_mask, _rna_dapi, _rot_dna_dapi, _dna_dapi
        else:
            return _dna_mask,

    def _save(self, save_hdf5_file:str)->None:
        if self.verbose:
            print(f"-- saving segmentation info from fov:{self.fov_id} into file: {save_hdf5_file}")
        with h5py.File(save_hdf5_file,'a') as _f:
            _fov_group = _f.require_group(str(self.fov_id))
            _fov_group.attrs['fov_id'] = self.fov_id
            _fov_group.attrs['fov_name'] = self.fov_name
            # add dataset:
            if 'dna_mask' in _fov_group.keys() and self.overwrite:
                del(_fov_group['dna_mask'])
            if 'dna_mask' not in _fov_group.keys():
                _mask_dataset = _fov_group.create_dataset('dna_mask', data=self.dna_mask)
            # add uid info
            _uid_group = _fov_group.require_group('cell_2_uid')
            for (_fov_id, _cell_id), _uid in self.fovcell_2_uid.items():
                if str(_cell_id) in _uid_group.keys() and self.overwrite:
                    del(_uid_group[str(_cell_id)])
                if str(_cell_id) not in _uid_group.keys():
                    _uid_group.create_dataset(str(_cell_id), data=_uid, shape=(1,))
        return

    def _load(self, save_hdf5_file:str)->bool:
        # load DNA
        _dna_mparam = _read_microscope_json(self.dna_microscope_file)
        _, _fov_id, _fov_name = self._load_dna_info(self.dna_save_file, _dna_mparam)
        self.fov_id = _fov_id
        self.fov_name = _fov_name
        if self.verbose:
            print(f"-- loading segmentation info from fov:{self.fov_id} into file: {save_hdf5_file}")
        if not os.path.exists(save_hdf5_file):
            print(f"--- sav_hdf5_file:{save_hdf5_file} does not exist, skip loading.")
            return False
        # load
        with h5py.File(save_hdf5_file, 'r') as _f:
            if str(self.fov_id) not in _f.keys():
                return False
            _fov_group = _f[str(self.fov_id)]
            # mask
            self.dna_mask = _fov_group['dna_mask'][:]
            # uid
            self.fovcell_2_uid = {}
            _uid_group = _fov_group['cell_2_uid']
            for _cell_id in _uid_group.keys():
                self.fovcell_2_uid[(self.fov_id, int(_cell_id))] = _uid_group[_cell_id][:][0]
        return True


def translate_segmentation(dapi_before, dapi_after, before_to_after_rotation,
                           label_before=None, label_after=None,
                           return_new_dapi=False,
                           verbose=True,
                           ):
    """ """
    from ..correction_tools.alignment import calculate_translation
    from ..correction_tools.translate import warp_3d_image
    # calculate drift
    _rot_dapi_after, _rot, _dft = calculate_translation(dapi_before, dapi_after, before_to_after_rotation,)
    # get dimensions
    _dz,_dx,_dy = np.shape(dapi_before)
    _rotation_angle = np.arcsin(_rot[0,1])/math.pi*180
    
    if label_before is not None:
        _seg_labels = np.array(label_before)
        _rotation_angle = -1 * _rotation_angle
        _dft = -1 * _dft
    elif label_after is not None:
        _seg_labels = np.array(label_after)
    else:
        ValueError('Either label_before or label_after should be given!')
    # generate rotation matrix in cv2
    if verbose:
        print('- generate rotation matrix')
    _rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1)
    # rotate segmentation
    if verbose:
        print('- rotate segmentation label with rotation matrix')
    _rot_seg_labels = np.array(
        [cv2.warpAffine(_seg_layer,
                        _rotation_M, 
                        _seg_layer.shape, 
                        flags=cv2.INTER_NEAREST,
                        #borderMode=cv2.BORDER_CONSTANT,
                        borderMode=cv2.BORDER_REPLICATE,
                        #borderValue=int(np.min(_seg_labels)
                        )
            for _seg_layer in _seg_labels]
        )
    # warp the segmentation label by drift
    _dft_rot_seg_labels = warp_3d_image(_rot_seg_labels, _dft, 
        warp_order=0, border_mode='nearest')
    if return_new_dapi:
        return _dft_rot_seg_labels, _rot_dapi_after
    else:
        return _dft_rot_seg_labels
    
# interpolate matrices
def interploate_z_masks(z_masks, 
                        z_coords, 
                        target_z_coords=default_dna_Zcoords,
                        mode='nearest',
                        verbose=True,
                        ):

    # target z
    _final_mask = []
    _final_coords = np.round(target_z_coords, 3)
    for _fz in _final_coords:
        if _fz in z_coords:
            _final_mask.append(z_masks[np.where(z_coords==_fz)[0][0]])
        else:
            if mode == 'nearest':
                _final_mask.append(z_masks[np.argmin(np.abs(z_coords-_fz))])
                continue
            # find nearest neighbors
            if np.sum(z_coords > _fz) > 0:
                _upper_z = np.min(z_coords[z_coords > _fz])
            else:
                _upper_z = np.max(z_coords)
            if np.sum(z_coords < _fz) > 0:
                _lower_z = np.max(z_coords[z_coords < _fz])
            else:
                _lower_z = np.min(z_coords)

            if _upper_z == _lower_z:
                # copy the closest mask to extrapolate
                _final_mask.append(z_masks[np.where(z_coords==_upper_z)[0][0]])
            else:
                # interploate
                _upper_mask = z_masks[np.where(z_coords==_upper_z)[0][0]].astype(np.float32)
                _lower_mask = z_masks[np.where(z_coords==_lower_z)[0][0]].astype(np.float32)
                _inter_mask = (_upper_z-_fz)/(_upper_z-_lower_z) * _lower_mask + (_fz-_lower_z)/(_upper_z-_lower_z) * _upper_mask
                _final_mask.append(_inter_mask)
                
    if verbose:
        print(f"- reconstruct {len(_final_mask)} layers")
    
    return np.array(_final_mask)