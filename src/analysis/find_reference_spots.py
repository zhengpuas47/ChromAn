import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import search_fovs_in_folders, Color_Usage
from analysis import AnalysisTask
from default_parameters import default_correction_folder

class FindRefenceSpots(AnalysisTask):
    """Class to find refence spots, as the centers of interest for downstream analysis.
    1. Load the first dax file as reference.
    2. Load channels with refence spots
    3. Apply bleedthrough, illumination and chromatic correction.
    4. 3D spot_finding to find the reference spots.
    5. Save the reference spots.
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
        # get ref id
        _ref_id = self.analysis_parameters.get('ref_id', 0) # by default, use the first one
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict()) # kewargs format
        _correction_params = {
            'corr_bleed': True,
            'corr_illumination': True,
            'corr_chromatic': True,
            'warp_image': True,
            'corr_drift': False,
            }
        _correction_params.update(self.analysis_parameters.get('correction_params', dict()))
        _fit_params = self.analysis_parameters.get('fitting_params', dict())
        if 'fit_channels' in _fit_params:
            del _fit_params['fit_channels']
        

        # scan subfolders
        _folders, _fovs = search_fovs_in_folders(self.data_folder)
        _fov_name = _fovs[self.field_of_view]
        # load color_usage to determine channels to be processed:
        _save_folder = self.analysis_parameters.get('save_folder', os.path.join(self.data_folder, 'Analysis'))        
        color_usage_df = Color_Usage(os.path.join(_save_folder, self.color_usage))
        # get ref_folder
        _ref_folder = os.path.join(self.data_folder, color_usage_df.index[_ref_id])
        self.ref_filename = os.path.join(_ref_folder, _fov_name)
        # find data_containing channels in this round:
        _valid_channels = color_usage_df.get_valid_channels(color_usage_df, hyb=color_usage_df.index[_ref_id], )
        _imaged_channels = color_usage_df.get_imaged_channels(color_usage_df, hyb=color_usage_df.index[_ref_id], )
        # output:
        _save_filename = os.path.join(_save_folder, f"{os.path.basename(self.ref_filename).split(os.pathsep)[0]}_{self.field_of_view}.hdf5")
        # now load dax with DaxProcesser:
        daxp = DaxProcesser(
            self.ref_filename,
            CorrectionFolder=self.analysis_parameters.get('correction_folder', default_correction_folder),
            Channels=_imaged_channels,
            **_loading_params,
            SaveFilename=_save_filename,
            FiducialChannel=color_usage_df.get_fiducial_channel(color_usage_df),
            )
        # load image
        daxp._load_image(sel_channels=_imaged_channels)
        # apply correction
        daxp.RunCorrection(**_correction_params)
        # fit spots
        daxp._fit_3D_spots(fit_channels=_valid_channels, **_fit_params)
        # save im and spots into hdf5:
        daxp._save_base_to_hdf5()
        for _ch in _valid_channels:
            if self.analysis_parameters.get('save_im', True):
                daxp._save_data_to_hdf5(_ch, save_type='im')
            # spots is definitely saving:
            daxp._save_data_to_hdf5(_ch, save_type='spots')
        
        if self.analysis_parameters.get('verbose', True):
            print("spots saved to hdf5.")

        return
            
## main
if __name__ == '__main__':
    # run
    task = FindRefenceSpots()

    task._run_analysis()
