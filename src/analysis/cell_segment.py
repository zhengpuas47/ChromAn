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
        