import numpy as np
import sys, os, h5py, time
import cv2
import logging
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# load image
from analysis import AnalysisTask
# Cellpose 
from segmentation_tools.segment import cellposeSegment

_default_segmentation_kwargs = {
    'min_size':1000,
    'do_3D': True,
    'cellprob_threshold': -3,
    'batch_size':24,
    'downsample': 5,
}
#_default_correction_folder = r'/lab/weissman_imaging/puzheng/Corrections/20240401-Merscope01_s11_n1200'
#_default_microscope_params = r'/lab/weissman_imaging/puzheng/Softwares/Weissman_MERFISH_Scripts/merlin_parameters/microscope/merscope01_microscope.json'

_default_analysis_parameters = {
    'channel_names': ['PolyT', 'DAPI'],
    'segment_nuclei': True,
    'segment_cytoplasm': True,
    'watershed': False,
    'report_matched': False,
    'report_nuclei': True,
    'report_cytoplasm': True,
    'report_watershed': False,
    'save_tiff': True,
    'save_masks': False,
}

    
class CellSegmentTask(AnalysisTask):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
        logger = logging.getLogger(__name__)
        logger.info("* ChromAn *")
        logger.info("Initializing Cell3DSegmentTask")
        self._load_analysis_parameters()
        logger.info("Loading parameters")
        # check if the input is a directory
        if not hasattr(self, 'data_folder') or self.data_folder is None:
            raise ValueError("Input data folder is required.")
        elif not os.path.isdir(self.data_folder):
            raise ValueError(f"Input data folder {self.data_folder} does not exist.")
        logger.info(f"Input directory: {self.data_folder}")
        # check save_folder:
        if not hasattr(self, 'save_folder') or self.save_folder is None:
            print(dir(self))
            print(self.color_usage)
            
            raise ValueError("Output data folder is required.")
        elif not os.path.isdir(self.save_folder):
            logger.info(f"Output data folder {self.save_folder} does not exist, creating it.")
            os.makedirs(self.save_folder)
        logger.info(f"Output directory: {self.save_folder}")
        # update kwargs
        self.segmentation_kwargs = _default_segmentation_kwargs.copy()
        self.segmentation_kwargs.update(kwargs)

        # check correction folder
        logger.info(f"Checking correction folder: {self.correction_folder}")
        # check microscope parameters
        logger.info(f"Checking microscope parameters: {self.microscope_params}")
        # check if color_usage is provided
        if not hasattr(self, 'color_usage') or self.color_usage is None:
            raise ValueError("Color usage file is required.")
        elif not os.path.isfile(self.color_usage):
            raise FileExistsError(f"Color usage file {self.color_usage} does not exist.")
        logger.info(f"Color usage file: {self.color_usage}")
        # check if FOV is provided
        if not hasattr(self, 'field_of_view') or self.field_of_view is None:
            raise ValueError("field_of_view is required.")
        elif not isinstance(self.field_of_view, int) or self.field_of_view < 0:
            raise ValueError(f"field_of_view should be an non-negative integer, got {type(self.field_of_view)} instead.")            
        logger.info(f"Field of view: {self.field_of_view}")
        # check analysis parameters:
        if not hasattr(self, 'analysis_parameters') or self.analysis_parameters == dict():
            logger.info("No analysis parameters provided, using default parameters.")
            self.analysis_parameters = _default_analysis_parameters.copy()
        else:
            logger.info(f"Using analysis parameters: {self.analysis_parameters}")
            # fill the missing parameters with default values
            for key, value in _default_analysis_parameters.items():
                if key not in self.analysis_parameters:
                    self.analysis_parameters[key] = value
        # assign channel names
        self.channel_names = self.analysis_parameters.get('channel_names', ['PolyT', 'DAPI'])
    
    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("Running CellSegmentTask")
        
        # create segmentation_class
        sg = cellposeSegment(
            data_folder=self.data_folder,
            save_folder=self.save_folder,
            field_of_view=self.field_of_view,
            channels=self.channel_names,
            correction_folder=self.correction_folder,
            color_usage=self.color_usage,
            microscope_params=self.microscope_params,
        )
        # load image:
        if self.correction_folder is None:
            _corr_illumination = False
            logger.info("Loading images without illumination correction.")
        else:
            _corr_illumination = True
            logger.info("Loading images with illumination correction.")
        sg.load_images(corr_illumination=_corr_illumination)
        # save tiff:
        if self.analysis_parameters.get('save_tiff', False):
            logger.info("Saving tiff files.")
            sg.save_tiff_images(save_folder=os.path.join(self.save_folder, 'segmentation_input_images'),)
        else:
            logger.info("Skipping saving tiff files.")
        # segment nuclei:
        if self.analysis_parameters.get('segment_nuclei', True):
            logger.info("Segmenting nuclei.")
            sg.segment_nuclei(
                min_size=self.segmentation_kwargs.get('min_size', 1000),
                do_3D=self.segmentation_kwargs.get('do_3D', True),
                cellprob_threshold=self.segmentation_kwargs.get('cellprob_threshold', -3),
                batch_size=self.segmentation_kwargs.get('batch_size', 24),
                downsample=self.segmentation_kwargs.get('downsample', 5),
                clear_border=self.segmentation_kwargs.get('clear_border', False),
            )
        else:
            logger.info("Skipping nuclei segmentation.")
        # segment cytoplasm:
        if self.analysis_parameters.get('segment_cytoplasm', True):
            logger.info("Segmenting cytoplasm.")
            sg.segment_cytoplasm(
                min_size=self.segmentation_kwargs.get('min_size', 1000),
                do_3D=self.segmentation_kwargs.get('do_3D', True),
                cellprob_threshold=self.segmentation_kwargs.get('cellprob_threshold', -3),
                batch_size=self.segmentation_kwargs.get('batch_size', 24),
                downsample=self.segmentation_kwargs.get('downsample', 5),
                clear_border=self.segmentation_kwargs.get('clear_border', False),
            )
        else:
            logger.info("Skipping cytoplasm segmentation.")
        # watershed:
        if hasattr(sg, 'nuc_masks') and self.analysis_parameters.get('watershed', False):
            logger.info("Applying watershed segmentation.")
            sg.watershed(
                waterline_th=self.segmentation_kwargs.get('waterline_th', 0.9),
                clear_border=self.segmentation_kwargs.get('clear_border', False),
            )
        # report segmentation results:
        if self.analysis_parameters.get('report_matched', False):
            # TODO: add matching script
            pass
        # report nuclei:
        if self.analysis_parameters.get('report_nuclei', True):
            logger.info("Reporting nuclei segmentation results.")
            if self.analysis_parameters.get('save_masks', False):
                logger.info("Saving nuclei masks.")
                sg.save_masks(
                    'nuc',
                    save_folder=os.path.join(self.save_folder, 'nuclei_masks'),
                    )
            logger.info("Saving nuclei polygons.")
            sg.save_polygons(
                'nuc',
                tolerance=self.analysis_parameters.get('tolerance', 0.01),
                save_folder=os.path.join(self.save_folder, 'nuclei_polygons'),
            )
        else:
            logger.info("Skipping nuclei reporting.")
        # report cytoplasm:
        if self.analysis_parameters.get('report_cytoplasm', True):
            logger.info("Reporting cytoplasm segmentation results.")
            if self.analysis_parameters.get('save_masks', False):
                logger.info("Saving cytoplasm masks.")
                sg.save_masks(
                    'cyto',
                    save_folder=os.path.join(self.save_folder, 'cytoplasm_masks'),
                )
            logger.info("Saving cytoplasm polygons.")
            sg.save_polygons(
                'cyto',
                tolerance=self.analysis_parameters.get('tolerance', 0.01),
                save_folder=os.path.join(self.save_folder, 'cytoplasm_polygons'),
            )
        else:
            logger.info("Skipping cytoplasm reporting.")
        # save watershed:
        if hasattr(sg, 'watershed_masks') and self.analysis_parameters.get('report_watershed', False):
            logger.info("Reporting watershed segmentation results.")
            if self.analysis_parameters.get('save_masks', False):
                logger.info("Saving watershed masks.")
                sg.save_masks(
                    'watershed',
                    save_folder=os.path.join(self.save_folder, 'watershed_masks'),
                )
            logger.info("Saving watershed polygons.")
            sg.save_polygons(
                'watershed',
                tolerance=self.analysis_parameters.get('tolerance', 0.01),
                save_folder=os.path.join(self.save_folder, 'watershed_polygons'),
            )
        else:
            logger.info("Skipping watershed reporting.")
        
        # matched?
        if self.analysis_parameters.get('report_matched', False) and hasattr(sg, 'cyto_masks') and hasattr(sg, 'nuc_masks'):
            logger.info("Reporting matched segmentation results.")
            # match
            sg.merged_masks = sg.match_cyto_nuclei_masks(
                sg.cyto_masks,
                sg.nuc_masks,
                overlap_threshold=self.analysis_parameters.get('overlap_threshold', 0.9),
            )
            merged_cyto_masks = np.abs(sg.merged_masks.copy())
            merged_nuc_masks = -1 * sg.merged_masks.copy()
            merged_nuc_masks[merged_nuc_masks < 0] = 0
            # assign:
            sg.nuc_masks = merged_nuc_masks
            sg.cyto_masks = merged_cyto_masks
            # save masks:
            if self.analysis_parameters.get('save_masks', False):
                logger.info("Saving merged masks.")
                sg.save_masks(
                    'merged',
                    save_folder=os.path.join(self.save_folder, 'merged_masks'),
                )
            # save polygons but this time to a different folder:
            logger.info("Saving merged polygons.")
            sg.save_polygons(
                'cyto',
                tolerance=self.analysis_parameters.get('tolerance', 0.01),
                save_folder=os.path.join(self.save_folder, 'merged', 'cytoplasm_polygons'),
            )
            sg.save_polygons(
                'nuc',
                tolerance=self.analysis_parameters.get('tolerance', 0.01),
                save_folder=os.path.join(self.save_folder, 'merged', 'nuclei_polygons'),
            )

if __name__ == '__main__':
    # run
    _task = CellSegmentTask()
    _task.run()