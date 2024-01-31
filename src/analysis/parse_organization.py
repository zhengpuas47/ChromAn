import numpy as np
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import Color_Usage, color_usage_kwds
from analysis import AnalysisTask

class ParseOrganizationTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__() 
        # by default, image_filename, color_usage, analysis_parameters, overwrite exists.
        # load analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        return
    
    def _run_analysis(self):
        # get params
        _ref_id = self.analysis_parameters.get('ref_id', 0) # by default, use the first one
        # load color_usage to determine channels to be processed:
        color_usage_df = Color_Usage(self.color_usage)
        # get ref_folder
        ref_folder = color_usage_df.iloc[_ref_id].index
        print(ref_folder)
        print(dir(self))
        
        
if __name__ == '__main__':
    # run
    task = ParseOrganizationTask()
    
    task._run_analysis()
