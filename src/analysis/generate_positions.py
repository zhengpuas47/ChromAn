import sys, os
import numpy as np
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Load necessary modules
from file_io.dax_process import DaxProcesser
from file_io.data_organization import search_fovs_in_folders, Color_Usage
from analysis import AnalysisTask

class GeneratePositions(AnalysisTask):
    """Class to generate position file that required by Merlin"""
    
    def __init__(self):
        super().__init__()
        # by default: 
        # image_filename, color_usage, analysis_parameters, 
        # save_folder, correction_folder, overwrite, test exists.
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        return
    
    def _run_analysis(self):
        # get rounds and fovs:
        _folders, _fovs = search_fovs_in_folders(self.data_folder)
        # load color_usage to determine channels to be processed:
        _color_usage_folder = self.analysis_parameters.get('color_usage_folder', os.path.join(self.data_folder, 'Analysis'))
        color_usage_df = Color_Usage(os.path.join(_color_usage_folder, self.color_usage))
        
        # go to reference:
        reference_round = int(self.analysis_parameters.get('reference_round', 0))
        print('reference_round:', reference_round)
        reference_folder = os.path.join(self.data_folder, color_usage_df.iloc[reference_round].name)
        # search fovs:
        positions = []
        if os.path.exists(reference_folder):
            # loop through files within this folder
            for _fov in _fovs:
                daxp = DaxProcesser(os.path.join(reference_folder, _fov), verbose=False)
                positions.append(daxp._FindGlobalPosition(daxp.filename))
        
        positions = np.array(positions)        
        if getattr(self, 'save_folder', None) is not None:
            # save positions
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
            # save positions
            print('save positions to:', self.save_folder)
            np.savetxt(os.path.join(self.save_folder, 'positions.txt'), positions, fmt='%.2f', delimiter=',')
        
if __name__ == '__main__':
    # run
    _task = GeneratePositions()
    _task._run_analysis()
