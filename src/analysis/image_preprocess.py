import numpy as np
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import Color_Usage, color_usage_kwds
from analysis import AnalysisTask


class ImagePreprocessTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__()
        # load analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict()) # kewargs format
        _fitting_params = self.analysis_parameters.get('fitting_params', dict()) # kewargs format
        # get output
        _save_folder = self.analysis_parameters.get('save_folder', os.path.dirname(self.image_filename))
        _save_filename = os.path.join(_save_folder,
            os.path.basename(self.image_filename).replace(os.extsep+self.image_filename.split(os.extsep)[-1], os.extsep+'hdf5'))
        # load color_usage to determine channels to be processed:
        color_usage_df = Color_Usage(self.color_usage)
        hyb_folder = os.path.basename(os.path.dirname(self.image_filename))
        # update parameters if not specified:
        if 'FiducialChannel' not in _loading_params:
            _loading_params['FiducialChannel'] = color_usage_df.get_fiducial_channel(color_usage_df)
        if 'DapiChannel' not in _loading_params:
            _loading_params['DapiChannel'] = color_usage_df.get_dapi_channel(color_usage_df)
        # determine fitting channels 
        if 'fit_channels' not in _fitting_params:
            _fitting_params['fit_channels'] = color_usage_df.get_valid_channels(color_usage_df, hyb_folder)
        # TODO: deterimine correction channels in a smarter way. For now I just decode all.
           
        # load image
        daxp = DaxProcesser(
            self.image_filename,
            **_loading_params,
            SaveFilename=_save_filename,
            )
        # load image
        daxp._load_image()
        # apply correction
        ## TODO: add corrections
        # save spots and raw_images
        daxp._save_base_to_hdf5()
        # fit spots
        daxp._fit_3D_spots(**_fitting_params)
        # save spots and image
        for _channel in daxp.channels:
            daxp._save_data_to_hdf5(channel=_channel, save_type='im')
            if hasattr(daxp, f"spots_{_channel}"):
                daxp._save_data_to_hdf5(channel=_channel, save_type='spots')
        
        return
        
if __name__ == '__main__':
    # run
    _task = ImagePreprocessTask()
    
    