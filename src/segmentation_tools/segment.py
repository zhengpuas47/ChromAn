# import enough packages for cellpose:
import numpy as np
import os, sys, time
import json
import logging
import skimage as ski
import fishtank as ft
import geopandas as gpd
from copy import copy
# cellpose default:
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/lab/weissman_imaging/puzheng/Softwares/"

# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from skimage.segmentation import watershed
from skimage.transform import resize
from tifffile import imwrite

# create a class to do cellpose segmentation:
# import abstract class:
from file_io.dax_process import DaxProcesser
from file_io.data_organization import Color_Usage, find_zfill_number

class cellposeSegment():
    """Cellpose segmentation class.
    Parameters:
    ----------
    data_folder : str
        Input file path or directory.
    field_of_view : int
        Field of view to process.
    save_folder : str
        Output file path.
    channels : list
        Channel for segmentation (e.g., DAPI or PolyT,DAPI).
    file_pattern : str
        Naming pattern for image files.
    correction_folder : str, optional
        Path to image corrections directory.
    color_usage : str
        Path to color usage file.
    """
    def __init__(self, 
                 data_folder: str, 
                 field_of_view: int, 
                 save_folder: str, 
                 channels: list, 
                 file_pattern: str = "{series}/Conv_zscan_{fov}.dax", 
                 correction_folder: str = None, 
                 color_usage: str = None,
                 microscope_params: str = None,
                 ):
        
        self.data_folder = data_folder
        self.field_of_view = field_of_view
        
        self.save_folder = save_folder
        
        self.channels = channels
        self.file_pattern = file_pattern
        self.correction_folder = correction_folder
        self.color_usage = color_usage
        # load color_usage:
        if os.path.isfile(self.color_usage):
            self.color_usage = Color_Usage(self.color_usage)
            self.fiducial_channel = self.color_usage.get_fiducial_channel(self.color_usage)
        else:
            raise FileNotFoundError(f"Color usage file {self.color_usage} not found.")
        # load microscope json:
        if isinstance(microscope_params, str) and os.path.isfile(microscope_params):
            print(f"Loading microscope json file {microscope_params}.")
            self.microscope_params = json.load(open(microscope_params, 'r'))
        elif isinstance(microscope_params, dict):
            self.microscope_params = microscope_params
        else:
            Warning(f"Microscope json file {microscope_params} not found.")
        
    def load_images(self,
        corr_illumination: bool = True,
        ):
        # load round 0 as ref:
        ref_name = self.color_usage.iloc[0].name
        num_digits = find_zfill_number(os.path.join(self.data_folder, ref_name))
        ref_filename = os.path.join(
            self.data_folder,
            self.file_pattern.format(series=ref_name, fov=str(self.field_of_view).zfill(num_digits)))
        ref_daxp = DaxProcesser(ref_filename, 
                                CorrectionFolder=self.correction_folder,
                                FiducialChannel=self.fiducial_channel)
        ref_daxp._load_image(sel_channels=[self.fiducial_channel])
        if corr_illumination:
            ref_daxp._corr_illumination()
        # TODO add z_offsets info:
        
        # save a few parameters:
        #self.z_offsets = ref_daxp.get_z_offsets() # TODO
        self.stage_position = ref_daxp._FindGlobalPosition(ref_daxp.filename)
        # init
        self.cyto_image, self.nuc_image = None, None
        # try to load if specified:
        cyto_names = ['PolyT','membrane','cyto']
        for _cyto_name in cyto_names:
            if _cyto_name in self.channels:
                # load info from color_usage:
                _cyto_info = self.color_usage.get_info(_cyto_name)[0]
                _cyto_filename = os.path.join(
                    self.data_folder,
                    self.file_pattern.format(series=_cyto_info['series'], fov=str(self.field_of_view).zfill(num_digits)))
                # load image:
                _cyto_daxp = DaxProcesser(_cyto_filename, 
                                         CorrectionFolder=self.correction_folder,
                                         FiducialChannel=self.fiducial_channel)
                _cyto_daxp._load_image(sel_channels=[_cyto_info['channel'], self.fiducial_channel])
                if corr_illumination:
                    _cyto_daxp._corr_illumination()
                _cyto_z_offsets = _cyto_daxp._FindChannelZpositions(_cyto_daxp.xml_filename)[_cyto_info['channel']]
                # calculate drift:
                _cyto_daxp._calculate_drift(getattr(ref_daxp, f"im_{self.fiducial_channel}"),)
                # apply drift to cyto image:
                _cyto_daxp._corr_warpping_drift_chromatic(correction_channels=[_cyto_info['channel']],
                                                          corr_chromatic=False)
                # apply param:
                _cyto_daxp._transform_by_microscope_param(correction_channels=[_cyto_info['channel']],
                                                          microscope_params=self.microscope_params)
                # save cyto image:
                self.cyto_image = getattr(_cyto_daxp, f"im_{_cyto_info['channel']}")
                self.cyto_z_offsets = _cyto_z_offsets
                break
        # do the same for nuclei:
        nuc_names = ['DAPI','nuclei']
        for _nuc_name in nuc_names:
            if _nuc_name in self.channels:
                # load info from color_usage:
                _nuc_info = self.color_usage.get_info(_nuc_name)[0]
                _nuc_filename = os.path.join(
                    self.data_folder,
                    self.file_pattern.format(series=_nuc_info['series'], fov=str(self.field_of_view).zfill(num_digits)))
                # load image:
                _nuc_daxp = DaxProcesser(_nuc_filename, 
                                         CorrectionFolder=self.correction_folder,
                                         FiducialChannel=self.fiducial_channel)
                _nuc_daxp._load_image(sel_channels=[_nuc_info['channel'], self.fiducial_channel])
                if corr_illumination:
                    _nuc_daxp._corr_illumination()
                _nuc_z_offsets = _nuc_daxp._FindChannelZpositions(_nuc_daxp.xml_filename)[_nuc_info['channel']]
                if _nuc_info['series'] != ref_name:
                    # calculate drift:
                    _nuc_daxp._calculate_drift(getattr(ref_daxp, f"im_{self.fiducial_channel}"),)
                    # apply drift to cyto image:
                    _nuc_daxp._corr_warpping_drift_chromatic(correction_channels=[_nuc_info['channel']],
                                                            corr_chromatic=False)
                # apply param:
                _nuc_daxp._transform_by_microscope_param(correction_channels=[_nuc_info['channel']],
                                                          microscope_params=self.microscope_params)
                # save cyto image:
                self.nuc_image = getattr(_nuc_daxp, f"im_{_nuc_info['channel']}")
                self.nuc_z_offsets = _nuc_z_offsets
                break
        # check if cyto and nuc images are loaded:
        if self.cyto_image is None:
            raise Warning(f"Cyto image not found in {self.channels}.")
        if self.nuc_image is None:
            raise Warning(f"Nuc image not found in {self.channels}.")
        # return
        return self.cyto_image, self.nuc_image
    
    def segment_nuclei(self,
        filter: str = None,
        filter_args: dict = None,
        gpu: bool = True,
        channels: list = [0, 0],
        diameter: int = None,
        batch_size: int = 32,
        cellprob_threshold: float = -2,
        downsample: float = 4.,
        do_3D: bool = True,
        min_size: int = 1000,
        clear_border: bool = True,
        model_args: dict = {},
        ):
        # run cellpose segmentation:
        """   
    diameter : int, optional
        Cellpose diameter.
    cellprob_threshold : float, optional
        Cell probability threshold.
    downsample : int, optional
        Downsampling factor.
    deconwolf : bool, optional
        Use DeconWolf for deconvolution.
    do_3D : bool, optional
        Use 3D segmentation.
    min_size : int, optional
        Minimum size of objects to keep.
    model_args : dict, optional
        Additional model arguments (e.g., key1=val1,key2=val2). 
        """
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Running nuclei segmentation by cellpose.")
        # import cellpose:
        from cellpose import models
        # check images:
        if not hasattr(self, 'nuc_image'):
            raise AttributeError("Nuc image not found. Please load images first.")
        else:
            img = self.nuc_image
        # Apply filter
        if filter is not None:
            ## TODO: fix bugs in here:
            if hasattr(ft.filters, filter):
                logger.info(f"Applying {filter} filter from fishtank")
                if filter == "deconwolf":
                    filter_args["gpu"] = gpu
                    filter_args["z_step"] = round(self.nuc_z_offsets[1] - self.nuc_z_offsets[0], 3)
                    filter_args["colors"] = ['405']
                img = getattr(ft.filters, filter)(img, **filter_args)
            elif hasattr(ski.filters, filter):
                logger.info(f"Applying {filter} filter from skimage")
                if len(channels) > 1:
                    filter_args["channel_axis"] = 0
                img = getattr(ski.filters, filter)(img, **filter_args)
            else:
                raise ValueError(f"{filter} filter found in fishtank.filters or skimage.filters.")
        # down-sample:
        if downsample is not None and downsample > 1:
            logger.info(f"Downsampling image by a factor of {downsample}")
            downscale = (1,) * (img.ndim - 2) + (downsample, downsample)
            img = ski.transform.downscale_local_mean(img, downscale)
        else:
            logger.info("No downsampling applied.")
            downsample = 1
            downscale = (1,) * img.ndim
            
        # run cellpose:
        logger.info(f"Running cellpose, GPU: {gpu}, diameter: {diameter}, cellprob_threshold: {cellprob_threshold}")
        _model = models.CellposeModel(gpu=gpu)
        
        zstep = self.nuc_z_offsets[1] - self.nuc_z_offsets[0] # um
        pixel_size = self.microscope_params['microns_per_pixel'] # um
        # check if diameter is None:
        if diameter is None:
            effective_diameter = None
        else:
            effective_diameter = diameter / downsample
        # evaluate:
        masks = _model.eval(
            img[np.newaxis, :, :, :],
            channel_axis=0,
            z_axis=1,
            anisotropy= zstep/(pixel_size * downsample),
            batch_size=batch_size,
            diameter=effective_diameter,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D,
            min_size=min_size/downsample**2,
            flow3D_smooth=1,
            **model_args
        )[0]
        # Clear border
        if clear_border:
            logger.info("Clearing border")
            masks = ski.segmentation.clear_border(np.pad(masks, ((1, 1), (0, 0), (0, 0)), mode="constant"))[1:-1, :, :]
        # scale back to original size:
        logger.info(f"Rescaling masks by a factor of {downsample} and reverse anisotropy {(pixel_size * downsample)/zstep}")
        #rescale = ((pixel_size * downsample)/zstep,) * (img.ndim - 2) + (downsample, downsample)
        masks = resize(masks, self.nuc_image.shape, order=0, preserve_range=True, )
        masks = np.array(masks, dtype=np.int32)
        # add:
        self.nuc_masks = masks
        # return mask:
        return self.nuc_masks
    
    def segment_cytoplasm(self,
        filter: str = None,
        filter_args: dict = None,
        gpu: bool = True,
        channels: list = [0, 0],
        diameter: int = None,
        batch_size: int = 32,
        cellprob_threshold: float = -2,
        downsample: float = 4.,
        do_3D: bool = True,
        min_size: int = 1000,
        clear_border: bool = True,
        model_args: dict = {},
        ):
        # run cellpose segmentation:
        """
        
        """
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Running cytoplasmic segmentation by cellpose.")
        # import cellpose:
        from cellpose import models
        # check images:
        if not hasattr(self, 'nuc_image'):
            raise AttributeError("Nuc image not found. Please load images first.")
        elif not hasattr(self, 'cyto_image'):
            raise AttributeError("Cyto image not found. Please load images first.")
        else:
            imgs = np.stack([self.nuc_image, self.cyto_image],axis=0)
            logger.info(f"Shape of images: {imgs.shape}")
        # Apply filters
        for _iimg, _img in enumerate(imgs):
            if filter is not None:
                ## TODO: fix bugs in here:
                if hasattr(ft.filters, filter):
                    logger.info(f"Applying {filter} filter from fishtank")
                    if filter == "deconwolf":
                        filter_args["gpu"] = gpu
                        filter_args["z_step"] = round(self.nuc_z_offsets[1] - self.nuc_z_offsets[0], 3)
                        filter_args["colors"] = ['405']
                    imgs[_iimg] = getattr(ft.filters, filter)(_img, **filter_args)
                elif hasattr(ski.filters, filter):
                    logger.info(f"Applying {filter} filter from skimage")
                    if len(channels) > 1:
                        filter_args["channel_axis"] = 0
                    imgs[_iimg] = getattr(ski.filters, filter)(_img, **filter_args)
                else:
                    raise ValueError(f"{filter} filter found in fishtank.filters or skimage.filters.")
        # down-sample:
        if downsample is not None and downsample > 1:
            logger.info(f"Downsampling image by a factor of {downsample}")
            downscale = (1,) * (imgs.ndim - 2) + (downsample, downsample)
            imgs = ski.transform.downscale_local_mean(imgs, downscale)
        else:
            logger.info("No downsampling applied.")
            downsample = 1
            downscale = (1,) * imgs.ndim
            
        # run cellpose:
        logger.info(f"Running cellpose, GPU: {gpu}, diameter: {diameter}, cellprob_threshold: {cellprob_threshold}")
        _model = models.CellposeModel(gpu=gpu)
        
        zstep = self.nuc_z_offsets[1] - self.nuc_z_offsets[0] # um
        pixel_size = 0.107 # um
        # check if diameter is None:
        if diameter is None:
            effective_diameter = None
        else:
            effective_diameter = diameter / downsample
        # evaluate:
        masks = _model.eval(
            imgs,
            channel_axis=0,
            z_axis=1,
            anisotropy= zstep/(pixel_size * downsample),
            batch_size=batch_size,
            diameter=effective_diameter,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D,
            min_size=min_size/downsample**2,
            flow3D_smooth=1,
            **model_args
        )[0]
        # Clear border
        if clear_border:
            logger.info("Clearing border")
            masks = ski.segmentation.clear_border(np.pad(masks, ((1, 1), (0, 0), (0, 0)), mode="constant"))[1:-1, :, :]
        # scale back to original size:
        logger.info(f"Rescaling masks by a factor of {downsample} and reverse anisotropy {(pixel_size * downsample)/zstep}")
        #rescale = ((pixel_size * downsample)/zstep,) * (imgs.ndim - 2) + (downsample, downsample)
        masks = resize(masks, self.nuc_image.shape, order=0, preserve_range=True, )
        masks = np.array(masks, dtype=np.int32)
        # add:
        self.cyto_masks = masks
        # return mask:
        return self.cyto_masks
    
    def watershed(self, 
        downsample: int = 1,
        filter: str = 'gaussian',
        filter_size: int = 3,
        filter_args: dict = {},
        waterline_th: float = 0.9,
        clear_border: bool = True,
        ):
        # run watershed segmentation:
        """
    mask : np.ndarray, optional
        Input mask for watershed segmentation.
    filter : str, optional  
        Filter to apply before watershed segmentation.
    filter_args : dict, optional
        Additional arguments for the filter.
        """
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Running watershed segmentation.")
        # check images:
        if not hasattr(self, 'nuc_masks'):
            raise AttributeError("Nuc masks not found. Please run cellpose first.")
        else:
            mask = copy(self.nuc_masks)
        # cyto image:
        if self.cyto_image is None:
            raise AttributeError("Cyto image not found. Please load images first.")
        else:   
            img = self.cyto_image
        # Apply filter
        if filter is not None:
            if filter == 'gaussian':
                logger.info(f"Applying {filter} filter from skimage")
                img = ski.filters.gaussian(img, sigma=filter_size, **filter_args)
            else:
                raise NotImplementedError(f"{filter} filter not implemented.")
        # down-sample:
        if downsample is not None and downsample > 1:
            logger.info(f"Downsampling image by a factor of {downsample}")
            downscale = (1,) * (img.ndim - 2) + (downsample, downsample)
            img = ski.transform.downscale_local_mean(img, downscale)
            # downsample mask:
            mask = ski.transform.downscale_local_mean(mask, downscale)
        else:
            logger.info("No downsampling applied.")
            downscale = (1,) * img.ndim
            downsample = 1
        # run watershed:
        logger.info(f"Generate watershed image with waterline threshold:{waterline_th}")
        watershed_im = 1 - (img - np.min(img)) / (np.max(img)-np.min(img))
        logger.info(f"Running watershed segmentation")
        watershed_masks = watershed(watershed_im, mask, mask=watershed_im < waterline_th)
        # scale back to original size:
        if downsample > 1:
            logger.info(f"Rescaling watershed masks by a factor of {downsample}")
            watershed_masks = ski.transform.rescale(watershed_masks, downscale, order=0, preserve_range=True, )
            watershed_masks = np.array(watershed_masks, dtype=np.int32)
        if clear_border:
            logger.info("Clearing border")
            watershed_masks = ski.segmentation.clear_border(np.pad(watershed_masks, ((1, 1), (0, 0), (0, 0)), 
                                                                   mode="constant"))[1:-1, :, :]
        
        # return:
        self.watershed_masks = watershed_masks
        self.watershed_z_offsets = self.nuc_z_offsets
        return self.watershed_masks
    
    def save_tiff_images(self, 
        save_folder: str = None,
        save_filename: str = None,
        ):
        # save tiff images:
        """
        Function to save tiff images.
        Parameters
        ----------
        save_folder : str, optional
            Folder to save images to.
        mask_type : str, optional
            Type of masks to save
        """
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Saving tiff images.")
        # check images:
        if not hasattr(self, 'cyto_image') and not hasattr(self, 'nuc_image'):
            raise AttributeError("Cyto and Nuc images not found. Please load images first.")
        
        imgs = [getattr(self, 'cyto_image'), getattr(self, 'nuc_image')]
        # remove empty images:
        imgs = [img for img in imgs if img is not None]
        
        # check save folder:
        if save_folder is None:
            save_folder = self.save_folder
        if not os.path.exists(save_folder):
            logger.info(f"Creating folder {save_folder}")
            os.makedirs(save_folder)
        # check savefile:
        if save_filename is None:
            save_filename = f"FOV{self.field_of_view}_segmentation_input_images.tif"
        if '.tif' not in save_filename and '.tiff' not in save_filename:
            save_filename = os.path.join(save_folder, f"{save_filename}.tif")
        else:
            save_filename = os.path.join(save_folder, save_filename)
        # save images:
        logger.info(f"Saving images to {save_filename}")
        imwrite(save_filename, np.stack(imgs, axis=0), metadata={'axes': 'CZYX'})

    def save_masks(self,
        mask_type:str,
        save_folder: str = None,
        save_filename: str = None,
        ):
        """Function to save masks to a file."""
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Saving masks.")
        # check masks:
        if mask_type == 'nuc':
            _masks = self.nuc_masks
        elif mask_type == 'cyto':
            _masks = self.cyto_masks
        elif mask_type == 'watershed':
            _masks = self.watershed_masks
        elif mask_type == 'merged':
            _masks = self.merged_masks
        else:
            raise ValueError(f"Unknown mask type: {mask_type}. Please use 'nuc', 'cyto', or 'watershed' or 'merged.")
        if _masks is None:
            raise AttributeError(f"{mask_type.capitalize()} masks not found. Please run segmentation first.")
        # check save folder:
        logger.info(f"Saving masks to folder: {save_folder}")
        if save_folder is None:
            save_folder = os.path.join(self.save_folder, f"{mask_type}_masks")
        if not os.path.exists(save_folder):
            logger.info(f"Creating folder {save_folder}")
            os.makedirs(save_folder)
        # check savefile:
        if save_filename is None:
            save_filename = f"masks_{self.field_of_view}.npy"
        if '.npy' not in save_filename:
            save_filename = os.path.join(save_folder, f"{save_filename}.npy")
        else:
            save_filename = os.path.join(save_folder, save_filename)
        # save masks:
        logger.info(f"Saving masks to {save_filename}")
        np.save(save_filename, _masks)

    def save_polygons(self,
        mask_type:str, 
        tolerance: float = 0.01,
        save_folder: str = None,
        save_filename: str = None,
        ):
        """Function to save polygons to a file.
        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            Polygons to save.
        save_folder : str, optional
            Folder to save polygons to.
        save_filename : str, optional
            Name of the file to save polygons to.
        """
        logger = logging.getLogger("cellpose")
        logger.info("* ChromAn *")
        logger.info("Convert mask to polygons.")
        if mask_type == 'nuc':
            _masks = self.nuc_masks
        elif mask_type == 'cyto':
            _masks = self.cyto_masks
        elif mask_type == 'watershed':
            _masks = self.watershed_masks
        else:
            raise ValueError(f"Unknown mask type: {mask_type}. Please use 'nuc', 'cyto', or 'watershed'.")
        if _masks is None:
            raise AttributeError(f"{mask_type.capitalize()} masks not found. Please run segmentation first.")
        # convert to polygons:
        polygons = self.masks_to_polygons(
            _masks, 
            tolerance=tolerance, 
            id='cell', 
            z='z',
        )
        # append features to polygons:
        polygons = self._append_polygon_features(
            polygons, 
            fov=self.field_of_view,
            z_offsets=getattr(self, f"{mask_type}_z_offsets", []), 
            stage_position=getattr(self, 'stage_position', [0, 0]),
            min_size=1000,
            id='cell',
        )
        # check save folder:
        logger.info(f"Saving polygons to folder: {save_folder}")
        if save_folder is None:
            save_folder = os.path.join(self.save_folder, f"{mask_type}_polygons")
        if not os.path.exists(save_folder):
            logger.info(f"Creating folder {save_folder}")
            os.makedirs(save_folder)
        # check savefile:
        if save_filename is None:
            save_filename = f"polygons_{self.field_of_view}.json"
        if '.json' not in save_filename and '.json' not in save_filename:
            save_filename = os.path.join(save_folder, f"{save_filename}.json")
        else:
            save_filename = os.path.join(save_folder, save_filename)
        # save polygons:
        logger.info(f"Saving polygons to {save_filename}")
        polygons.to_file(save_filename, driver='GeoJSON')
        
    @staticmethod
    def match_cyto_nuclei_masks(
        cyto_masks: np.ndarray,
        nuc_masks: np.ndarray,
        overlap_threshold: float = 0.9,
        ):
        """Function to match cytoplasm and nuclei masks."""
        logger = logging.getLogger("cellpose")
        logger.info("Matching cytoplasm and nuclei masks.")
        # check masks:
        if cyto_masks is None or nuc_masks is None:
            raise ValueError("Cytoplasm and nuclei masks are required for matching.")
        if cyto_masks.ndim != nuc_masks.ndim:
            raise ValueError("Cytoplasm and nuclei masks must have the same number of dimensions.")
        if cyto_masks.shape != nuc_masks.shape:
            raise ValueError("Cytoplasm and nuclei masks must have the same shape.")
        # match masks:
        logger.info(f"Matching masks with overlap threshold {overlap_threshold}")
        # calculate volume overlaps:
        merged_masks = np.zeros(np.shape(cyto_masks), dtype=np.int16)

        for _cyto in np.unique(cyto_masks):
            if _cyto == 0:
                continue
            masked_pixels = nuc_masks[cyto_masks == _cyto]
            if len(masked_pixels) < 100:
                continue
            matched_nuc, matched_nuc_counts = np.unique(masked_pixels, return_counts=True)
            # calculate fraction:
            matched_nuc_counts = matched_nuc_counts[matched_nuc > 0]
            matched_nuc = matched_nuc[matched_nuc > 0]
            if len(matched_nuc) == 0:
                continue
            overlaps = []
            for _nuc, _count in zip(matched_nuc, matched_nuc_counts):
                overlaps.append(_count / np.sum(nuc_masks==_nuc))
            overlaps = np.array(overlaps)
            if np.max(overlaps) > 0.9 and len(overlaps > 0.9) == 1:
                print(f"Found match for {_nuc}")
                # update merged_mask:
                merged_masks[cyto_masks == _cyto] = np.max(merged_masks)+1
                merged_masks[nuc_masks== matched_nuc[np.argmax(overlaps)]] = -1 * np.max(merged_masks)
            elif np.max(overlaps) <= 0.9:
                print("No good enough overlap, skip")
            elif len(overlaps>0.9) > 1:
                print("Multiple nuclei matched to one cyto")
            
        return merged_masks
            

    @staticmethod
    def masks_to_polygons(
        masks: np.ndarray,
        tolerance: float = 0.01,
        id: str = 'cell',
        z: str = 'z',
        ):
        # convert masks to polygons:
        """
        Function to convert masks to polygons.
        Parameters
        ----------
        type : str, optional
            Type of masks to convert,
        tolerance : float, optional
            Tolerance for polygon conversion.
        """
        logger = logging.getLogger("cellpose")
        # convert to polygons:
        logger.info(f"Converting masks to polygons with tolerance {tolerance}")
        polygons = ft.seg.masks_to_polygons(masks, tolerance=tolerance, id=id, z=z)
        return polygons

    @staticmethod
    def _append_polygon_features(
        polygons: gpd.GeoDataFrame, 
        fov: int,
        z_offsets: list, 
        stage_position: list,
        min_size: int = 1000,
        id: str = 'cell',
        ):
        """Append features to polygons.
        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            Polygons to append features to.
        features : dict
            Features to append to polygons.
        id : str, optional
            Name of the id column in the polygons GeoDataFrame.
        """
        logger = logging.getLogger("cellpose")
        logger.info("Converting masks to polygons.")
        # create zmapping
        z_mapping = {z: z_offsets[i] for i, z in enumerate(sorted(polygons["z"].unique()))}
        polygons["global_z"] = polygons["z"].map(z_mapping)
        # Calculate polygon area
        polygons.index = polygons[id].values
        polygons["area"] = polygons.area
        polygons["area"] = polygons.groupby(id)["area"].transform("sum")
        # Remove small cells
        logger.info(f"Filtering out small cells with size < {min_size}")
        polygons = cellposeSegment._remove_small_cells(polygons, min_size=min_size, id=id)
        # Save results
        logger.info(f"Adding FOV information")
        polygons["fov"] = fov
        polygons["x_offset"] = stage_position[0]
        polygons["y_offset"] = stage_position[1]
        return polygons
        
    @staticmethod
    def _remove_small_cells(
        polygons: gpd.GeoDataFrame, 
        min_size: int = 1000,
        id: str = 'cell',
        ):
        """Remove small cells from polygons.
        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            Polygons to remove small cells from.
        min_size : int, optional
            Minimum size of cells to keep.
        id : str, optional
            Name of the id column in the polygons GeoDataFrame.
        """
        if "area" not in polygons.columns:
            raise ValueError("Area column not found in polygons.")
        # Filter small cells
        n_before = polygons[id].nunique()
        polygons = polygons[polygons["area"] > min_size].copy()
        n_after = polygons[id].nunique()
        print(f"Removed {n_before - n_after} out of {n_before} total cells")
        return polygons