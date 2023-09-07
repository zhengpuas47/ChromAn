import numpy as np
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from file_io.dax_process import DaxProcesser
from file_io.data_organization import Color_Usage, color_usage_kwds
from analysis import AnalysisTask
# Cellpose 
from cellpose import models
import time

class CellSegmentTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__()
        # load analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        
        # load dapi image from the previous round:
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict()) # kewargs format
        _fitting_params = self.analysis_parameters.get('fitting_params', dict()) # kewargs format
 
         # get output from previous step:
        _save_folder = self.analysis_parameters.get('save_folder', os.path.dirname(self.image_filename))
        _save_filename = os.path.join(_save_folder,
            os.path.basename(self.image_filename).replace(os.extsep+self.image_filename.split(os.extsep)[-1], os.extsep+'hdf5'))
        # load color_usage to determine channels to be processed:
        color_usage_df = Color_Usage(self.color_usage)
        hyb_folder = os.path.basename(os.path.dirname(self.image_filename))
