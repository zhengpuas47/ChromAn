import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import search_fovs_in_folders, Color_Usage
from analysis import AnalysisTask
from default_parameters import default_correction_folder

class CheckOrganizationTask(AnalysisTask):
    """Step 1 analysis: check existence of all dax files listed in the color_usage file.
    1. Load the first dax file as reference.
    2. Load the fiducial channel and dapi channel.
    3. Apply illumination correction.
    4. Save the illumination corrected image.
    """
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
        if self.hyb_id == -1:
            _hyb_id = 0 # by default, use the first one
        else:
            _hyb_id = self.hyb_id
        print(f"Check ref hyb_id: {_hyb_id}")
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict()) # kewargs format
        # output:
        _save_folder = self.analysis_parameters.get('save_folder', os.path.join(self.data_folder, 'Analysis'))
        _save_filename = os.path.join(_save_folder, f"organization_{self.field_of_view}.hdf5")
        # scan subfolders
        _folders, _fovs = search_fovs_in_folders(self.data_folder)
        _fov_name = _fovs[self.field_of_view]
        # load color_usage to determine channels to be processed:
        _color_usage_folder = self.analysis_parameters.get('color_usage_folder', os.path.join(self.data_folder, 'Analysis'))
        color_usage_df = Color_Usage(os.path.join(_color_usage_folder, self.color_usage))
        # get ref_folder
        _ref_folder = os.path.join(self.data_folder, color_usage_df.index[_hyb_id])
        self.ref_filename = os.path.join(_ref_folder, _fov_name)
        # update parameters if not specified:
        if 'FiducialChannel' not in _loading_params:
            _loading_params['FiducialChannel'] = color_usage_df.get_fiducial_channel(color_usage_df)
        if 'DapiChannel' not in _loading_params:
            _loading_params['DapiChannel'] = color_usage_df.get_dapi_channel(color_usage_df)
        # load reference image
        daxp = DaxProcesser(
            self.ref_filename,
            CorrectionFolder=self.analysis_parameters.get('correction_folder', default_correction_folder),
            **_loading_params,
            SaveFilename=_save_filename,
            )
        # lodad image
        daxp._load_image(sel_channels=[_loading_params['FiducialChannel'], _loading_params['DapiChannel']])
        # apply correction
        daxp._corr_illumination(correction_channels=[_loading_params['FiducialChannel'], _loading_params['DapiChannel']])
        # save
        daxp._save_to_npy(save_channels=[_loading_params['FiducialChannel']],) # save fiducial image for later use
        daxp._save_to_npy(save_channels=[_loading_params['DapiChannel']],) # save dapi for segmentation loading
        
        # for all the other images, check existence:
        for _id in range(len(color_usage_df)):
            if _id == _hyb_id:
                continue
            _folder = os.path.join(self.data_folder, color_usage_df.index[_id])
            _filename = os.path.join(_folder, _fov_name)
            if not os.path.exists(_filename):
                raise FileExistsError(f"File {_filename} does not exist.")
            
        return
    
## main
if __name__ == '__main__':
    # run
    task = CheckOrganizationTask()
    
    task._run_analysis()
