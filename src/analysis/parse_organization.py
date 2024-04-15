import numpy as np
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import search_fovs_in_folders, Color_Usage
from analysis import AnalysisTask
from default_parameters import default_correction_folder

class ParseOrganizationTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__() 
        # by default, image_filename, color_usage, analysis_parameters, save_folder, correction_folder, overwrite, test exists.
        # load detailed analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        return
    
    def _run_analysis(self):
        # get params
        _ref_id = self.analysis_parameters.get('ref_id', 0) # by default, use the first one
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict()) # kewargs format
        # output:
        _save_folder = self.analysis_parameters.get('save_folder', os.path.join(self.data_folder, 'Analysis'))
        _save_filename = os.path.join(_save_folder, f"organization_{self.field_of_view}.hdf5")
        # scan subfolders
        _folders, _fovs = search_fovs_in_folders(self.data_folder)
        _fov_name = _fovs[self.field_of_view]
        # load color_usage to determine channels to be processed:
        color_usage_df = Color_Usage(self.color_usage)
        # get ref_folder
        _ref_folder = os.path.join(self.data_folder, color_usage_df.index[_ref_id])
        self.ref_filename = os.path.join(_ref_folder, _fov_name)
        print(self.ref_filename)
        # update parameters if not specified:
        if 'FiducialChannel' not in _loading_params:
            _loading_params['FiducialChannel'] = color_usage_df.get_fiducial_channel(color_usage_df)
        if 'DapiChannel' not in _loading_params:
            _loading_params['DapiChannel'] = color_usage_df.get_dapi_channel(color_usage_df)

        # load image
        daxp = DaxProcesser(
            self.ref_filename,
            CorrectionFolder=self.analysis_parameters.get('correction_folder', default_correction_folder),
            **_loading_params,
            SaveFilename=_save_filename,
            )
        # lodad image
        daxp._load_image(sel_channels=[_loading_params['FiducialChannel']])
        # apply correction
        daxp._corr_illumination(correction_channels=[_loading_params['FiducialChannel']])
        # save
        daxp._save_to_npy(save_channels=[_loading_params['FiducialChannel']],) 
        
        return
    
if __name__ == '__main__':
    # run
    task = ParseOrganizationTask()
    
    task._run_analysis()
