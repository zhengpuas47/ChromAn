import os, sys, re, time, h5py, pickle, sys
import numpy as np
import xml.etree.ElementTree as ET
from copy import copy
import json
from nd2 import ND2File
from skimage.registration import phase_cross_correlation
# import relative:
sys.path.append("..")
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# default params
#from default_parameters import *
from default_parameters import default_num_buffer_frames,default_num_empty_frames,default_channels,default_ref_channel,default_im_size,default_dapi_channel
# usful functions
from correction_tools.load_corrections import load_correction_profile
#from spot_tools.spot_class import Spots3D

class Nd2Processer(ND2File):

    def __init__(self,
                 ImageFilename:str,
                 CorrectionFolder=None,
                 FiducialChannel=None,
                 DapiChannel=405,
                 RefCorrectionChannel=640,
                 SaveFilename=None,
                 auto_load_image:bool = False,
                 nd2_args=[],
                 verbose=True,
                 *args,
                 **Kwargs,
                ):
        """"""
        super().__init__(ImageFilename, *nd2_args)
        # load metadata, required to determine channels:
        self._load_metadata()
        self.verbose = verbose
        # assign parameters:
        # save filename
        if SaveFilename is None:
            self.save_filename = os.path.join(
                os.path.dirname(self.path),
                os.path.basename(self.path).split('.nd2')[0] + '_processed.hdf5',
            )
        elif isinstance(SaveFilename, str) and SaveFilename != ImageFilename:
            self.save_filename = SaveFilename
        else:
            raise TypeError("SaveFilename should be a string of file full path.")
        self.saving_log = {}
        # Correction folder
        self.correction_folder = CorrectionFolder
        
        if FiducialChannel is not None and str(FiducialChannel) in self.channels:
            self.fiducial_channel = str(FiducialChannel)
        if DapiChannel is not None and str(DapiChannel) in self.channels:
            self.dapi_channel = str(DapiChannel)
        elif DapiChannel is None and default_dapi_channel in self.channels:
            self.dapi_channel = str(default_dapi_channel)
        if RefCorrectionChannel is not None and str(RefCorrectionChannel) in self.channels:
            self.ref_correction_channel = str(RefCorrectionChannel)
        elif RefCorrectionChannel is None and len(self.channels) > 1:
            self.ref_correction_channel = str(self.channels[1])
        # additional attributes:
            
        if auto_load_image:
            self._load_image()
        # Log for corrections:
        self.correction_log = {_ch:{} for _ch in self.channels}
        self.correction_params = {}
    
    def _load_metadata(self):
        self.open()
        self.image_metadata = copy(self.metadata)
        self.image_unstructured_metadata = copy(self.unstructured_metadata())
        self.close() 
        # load channels:
        self.channels = [str(int(re.search(r'[0-9]+', _ch.channel.name).group()))
            for _ch in self.metadata.channels]
        self.channel_indices = [int(_ch.channel.index) for _ch in self.metadata.channels]
        # 
        return
    
    def _load_image(self, return_images=False):
        # load metadata first:
        if not hasattr(self, 'channels') or not hasattr(self, 'channel_indices'):
            self._load_metadata()
        # load images:
        self.open()
        _images = self.asarray()[:]
        self.close()        
        # separate_images:
        for _i, (_ch, _ch_ind) in enumerate(zip(self.channels, self.channel_indices)):
            if self.metadata.channels[_i].loops.ZStackLoop == 0:
                setattr(self, f"im_{_ch}", _images[:,_ch_ind])
            elif self.metadata.channels[_i].loops.ZStackLoop is None:
                setattr(self, f"im_{_ch}", _images[_ch_ind])
        # return
        if return_images:
            return _images
    # Function to calculate drift by cross_correlation:
    def _calculate_drift(
        self,
        RefImage:np.ndarray,
        FiducialChannel=None,
        use_crossCorrelation=True,
        save_attrs:bool=True,
        save_ref_im:bool=True,
        precision_fold=100,
        overwrite:bool=False,
    ) -> np.ndarray:
        """Calculate drift given reference image
        """
        if hasattr(self, 'drift') and hasattr(self, 'drift_flag') and not overwrite:
            if self.verbose:
                print(f"- Drift already calculated, skip.")
            return self.drift, self.drift_flag
        # Load drift image
        if FiducialChannel is None and hasattr(self, 'fiducial_channel'):
            FiducialChannel = getattr(self, 'fiducial_channel')
        elif FiducialChannel is not None:
            FiducialChannel = str(FiducialChannel)
            if not hasattr(self, 'fiducial_channel'):
                self.fiducial_channel = FiducialChannel
        else:
            raise ValueError(f"Wrong input value for FiducialChannel: {FiducialChannel}")
        # save ref_im if specified
        if save_ref_im:
            self.ref_im = RefImage
        # make sure drift image exists:
        if not hasattr(self, f"im_{FiducialChannel}"):
            self._load_image()
        # align image:
        if use_crossCorrelation:
            _drift, _error, _phasediff = phase_cross_correlation(
                RefImage, 
                getattr(self, f"im_{FiducialChannel}"), 
                upsample_factor=precision_fold,
            )
            if self.verbose:
                print(f"Drift by phase cross correlation: {_drift}, {_error}, {_phasediff}")
            _drift_flag = 1
        else:
            from correction_tools.alignment import align_image
            # align image
            _drift, _drift_flag = align_image(
                getattr(self, f"im_{FiducialChannel}"),
                RefImage, 
                fiducial_channel=FiducialChannel,
                all_channels=self.channels,
                verbose=self.verbose, 
            )  
        # save attribute and return
        if save_attrs:
            # drift channel
            self.fiducial_channel = FiducialChannel
            # drift results
            self.drift = _drift
            self.drift_flag = _drift_flag
        # return
        return _drift, _drift_flag
    
    def _warp_image_by_drift(self,
            correction_channels=None,
            warp_order=3,
            border_mode='constant',
            save_attrs=True,
        ):
        """_summary_

        Args:
            warp_order (int, optional): _description_. Defaults to 3.
            border_mode (str, optional): _description_. Defaults to 'constant'.
            save_attrs (bool, optional): _description_. Defaults to True.
        """
        from correction_tools.translate import old_warp_3d_image
        corrected_images = []
        # check if images is loaded:
        if not hasattr(self, 'channels'):
            self._load_metadata()
        if not hasattr(self, f"im_{self.channels[0]}"):
            self._load_image()
        if not hasattr(self, 'drift'):
            raise AttributeError(f"No drift was provided for this nd2_process object, exit!")
        if correction_channels is None:
            correction_channels = self.channels
        for _ch in correction_channels:
            # if already warpped, dont do anything:
            if self.correction_log[_ch].get('warp', False):
                print(f"- Skip channel:{_ch} because it has been warpped.")
                if not save_attrs:
                    corrected_images.append(getattr(self, f"im_{_ch}"))
                continue
            else:
                _warped_im = old_warp_3d_image(
                    getattr(self, f'im_{_ch}'),
                    self.drift,
                    None,
                    warp_order=warp_order,
                    border_mode=border_mode,
                    verbose=self.verbose,
                )
                if save_attrs:
                    self.correction_log[_ch]['warp'] = True
                    self.correction_params['warp'] = {
                        _ch:{'warp_order':warp_order,'border_mode':border_mode,}
                    }
                    setattr(self, f"im_{_ch}", _warped_im)
                else:
                    corrected_images.append(_warped_im)
            
        if save_attrs:
            return
        else:  
            return corrected_images
        
    
    def _FindChannelZpositions(self):
        if not hasattr(self, 'channels') or not hasattr(self, '_expand_coords'):
            self._load_metadata()
        _z_coords = self._expand_coords()['Z']
        return {_ch:_z_coords for _ch in self.channels}
    
    def _GetImageSize(self):
        return dict(self.sizes)
    
    def _FindGlobalPosition(self):
        if not hasattr(self, 'image_unstructured_metadata'):
            self._load_metadata()
        _pos_metadata = self.image_unstructured_metadata['ImageMetadataSeqLV|0']['SLxPictureMetadata']
        # x,y pos:
        x_pos_key = [_k for _k in _pos_metadata.keys() if 'XPos' in _k][0]
        y_pos_key = [_k for _k in _pos_metadata.keys() if 'YPos' in _k][0]
        z_pos_key = [_k for _k in _pos_metadata.keys() if 'ZPos' in _k][0]
        
        global_pos_dict = {"X":_pos_metadata[x_pos_key],"Y":_pos_metadata[y_pos_key],"Z":_pos_metadata[z_pos_key],}
        return global_pos_dict
    
    