import numpy as np
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from analysis_input import analysis_input_parser
from file_io.dax_process import DaxProcesser
from analysis import AnalysisTask


class ImagePreprocessTask(AnalysisTask):
    def __init__(self):
        # inherit
        super().__init__()
        # load analysis_parameters
        self._load_analysis_parameters()
        print('parameters:', self.analysis_parameters)
        # get params
        _loading_params = self.analysis_parameters.get('loading_params', dict())
        _fitting_params = self.analysis_parameters.get('fitting_params', dict())
        # get output
        _save_folder = self.analysis_parameters.get('save_folder', os.path.dirname(self.image_filename))
        _save_filename = os.path.join(_save_folder,
            os.path.basename(self.image_filename).replace(os.extsep+self.image_filename.split(os.extsep)[-1], os.extsep+'hdf5'))
        # load image
        daxp = DaxProcesser(
            self.image_filename,
            *_loading_params,
            SaveFilename=_save_filename,
            )
        # load image
        daxp._load_image()
        # apply correction
        ## TODO: add corrections
        # fit spots
        daxp._fit_3D_spots(*_fitting_params)
        # save spots and raw_images
        
        return
        

if __name__ == '__main__':
    # run
    _task = ImagePreprocessTask()
    