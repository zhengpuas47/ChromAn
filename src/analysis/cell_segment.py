import numpy as np
import sys, os, h5py, time
import cv2
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import Color_Usage, color_usage_kwds
from analysis import AnalysisTask
# Cellpose 
from cellpose import models
from torch.cuda import empty_cache

_default_segmentation_kwargs = {
    'diameter':  200, #for in-gel
    'anisotropy': 4.673, #500nm/107nm, ratio of z-pixel vs x-y pixels
    'channels':[0,0],
    'min_size':4000,
    'do_3D': True,
    'cellprob_threshold': 0,
    'batch_size':25,
}

class Cell3DSegmentTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__()
        # load analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        
        # load dapi image from the previous round:
        # get params
        _cellpose_params = self.analysis_parameters.get('cellpose_params', dict()) # kewargs format 
        
         # get output from previous step:
        _save_folder = self.analysis_parameters.get('save_folder', os.path.dirname(self.image_filename))
        _save_filename = os.path.join(_save_folder,
            os.path.basename(self.image_filename).replace(os.extsep+self.image_filename.split(os.extsep)[-1], os.extsep+'hdf5'))
        # load color_usage to determine channels to be processed:
        color_usage_df = Color_Usage(self.color_usage)
        dapi_channel = color_usage_df.get_dapi_channel(color_usage_df)
        hyb_folder = os.path.basename(os.path.dirname(self.image_filename))
        ## default parameters:
        _label_key = 'dapi_mask'
        _overwrite = True # TODO:pass this from analysis_parameters
        _compression = 'gzip'
        #_overwrite = getattr(self, 'overwrite', True)
        # load label if already exist:
        if os.path.isfile(_save_filename):
            with h5py.File(_save_filename, 'r') as _f:
                if _label_key in _f and not _overwrite:
                    print(f"-- segmentation mask already exist, skip.")
                    return
        print(f"-- segmentation mask file doesn't exist, continue.")
        # load dapi image:
        # create class
        #daxp = DaxProcesser(self.image_filename, SaveFilename=_save_filename,) # keep the minimum
        # load image from saveifle:
        with h5py.File(_save_filename, 'r') as _f:
            _dapi_im_key = f"{hyb_folder}/{dapi_channel}/im"
            _dapi_im = _f[_dapi_im_key][:]
        # rescale
        rescale_factor = _cellpose_params.get("rescale_factor", 0.5) # by default, subsample 2 folds
        
        _input_dapi_im = np.array([cv2.resize(_ly, (int(_dapi_im.shape[-2]*rescale_factor),int(_dapi_im.shape[-1]*rescale_factor)) ) 
                                for _ly in _dapi_im])
        # cellpose_param:
        _use_gpu = _cellpose_params.get("gpu", True) 
        _model = _cellpose_params.get("model_type", 'nuclei') 
        _segmentation_kwargs = {
            _k: _cellpose_params.get(_k, _v)
            for _k,_v in _default_segmentation_kwargs.items()
        }
        # modify size based on rescaling:
        _segmentation_kwargs['diameter'] = _segmentation_kwargs['diameter'] * rescale_factor
        _segmentation_kwargs['anisotropy'] = _segmentation_kwargs['anisotropy'] * rescale_factor
        _segmentation_kwargs['min_size'] = _segmentation_kwargs['min_size'] * rescale_factor**2

        # Create cellpose model
        print(f"- run Cellpose segmentation", end=' ')
        _cellpose_start = time.time()
        empty_cache() # empty cache to create new model
        seg_model = models.CellposeModel(gpu=_use_gpu, model_type=_model)
        # Run cellpose prediction
        labels3d, _, _ = seg_model.eval(np.stack([_input_dapi_im,_input_dapi_im], axis=3), 
                                        ** _segmentation_kwargs,)
        # resize segmentation label back
        corr_labels3d = np.array([cv2.resize(_ly, _dapi_im.shape[-2:], 
                                        interpolation=cv2.INTER_NEAREST_EXACT) 
                            for _ly in labels3d])
        print(f"in {time.time()-_cellpose_start:.3f}s.")
        # save
        with h5py.File(_save_filename, 'a') as _f:
            # delete old if overwrite 
            if _label_key in _f and _overwrite:
                del _f[_label_key]
            if _label_key not in _f:
                # save
                print(f"-- saving segmentation labels into file: {_save_filename}, {_label_key}")
                _dataset = _f.require_dataset(_label_key, data=corr_labels3d,  
                                            shape=corr_labels3d.shape, 
                                            dtype=corr_labels3d.dtype,
                                            chunks=tuple([1, corr_labels3d.shape[-2],corr_labels3d.shape[-1]]),
                                            compression=_compression,
                                            )
            else:
                print("-- segmentation mask already exist, skip.")
        # return
        return 

if __name__ == '__main__':
    # run
    _task = Cell3DSegmentTask()
    