import sys, os
import numpy as np
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import search_fovs_in_folders, Color_Usage
from analysis import AnalysisTask
from default_parameters import default_correction_folder

class MatchingSpots(AnalysisTask):
    """Find matching spots between two channels."""
    def __init__(self):
        # inherit
        super().__init__() 
        # by default, image_filename, color_usage, analysis_parameters, save_folder, correction_folder, overwrite, test exists.
        # load detailed analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        return
    
    def _run_analysis(self):
        # load spots from hdf5 file:
        _hyb_id = self.analysis_parameters.get('hyb_id', 0) # by default, assume spots are from hyb 0
        
        _spots1_filename = self.analysis_parameters.get('spots1_filename', None)
        
###############

if __name__ == '__main__':
    # run
    _task = ImagePreprocessTask()
    _task._run_analysis()
