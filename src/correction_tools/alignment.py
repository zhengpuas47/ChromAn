import os, time
import numpy as np
from ..default_parameters import *

from..file_io.dax_process import load_image_base



def _find_boundary(_ct, _radius, _im_size):
    _bds = []
    for _c, _sz in zip(_ct, _im_size):
        _bds.append([max(_c-_radius, 0), min(_c+_radius, _sz)])
    
    return np.array(_bds, dtype=np.int32)


def generate_drift_crops(single_im_size=default_im_size, 
                         coord_sel=None, drift_size=None):
    """Function to generate drift crop from a selected center and given drift size
    keywards:
        single_im_size: single image size to generate crops, np.ndarray like;
        coord_sel: selected center coordinate to split image into 4 rectangles, np.ndarray like;
        drift_size: size of drift crop, int or np.int32;
    returns:
        crops: 4x3x2 np.ndarray. 
    """
    # check inputs
    _single_im_size = np.array(single_im_size)
    if coord_sel is None:
        coord_sel = np.array(_single_im_size/2, dtype=np.int32)
    if coord_sel[-2] >= _single_im_size[-2] or coord_sel[-1] >= _single_im_size[-1]:
        raise ValueError(f"wrong input coord_sel:{coord_sel}, should be smaller than single_im_size:{single_im_size}")
    if drift_size is None:
        drift_size = int(np.max(_single_im_size)/4)
        
    # generate crop centers
    crop_cts = [
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1],]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1],]),                               
    ]
    # generate boundaries
    crops = [_find_boundary(_ct, _radius=drift_size/2, _im_size=single_im_size) for _ct in crop_cts]
        
    return np.array(crops)


_default_align_corr_args={
    'single_im_size':default_im_size,
    'num_buffer_frames':default_num_buffer_frames,
    'num_empty_frames':default_num_empty_frames,
    'correction_folder':default_correction_folder,
    'illumination_corr':True,
    'bleed_corr': False, 
    'chromatic_corr': False,
    'z_shift_corr': False, 
    'hot_pixel_corr': True,
    'normalization': False,
}

_default_align_fitting_args={
    'th_seed': 300,
    'th_seed_per': 95, 
    'use_percentile': False,
    'use_dynamic_th': True,
    'min_dynamic_seeds': 10,
    'max_num_seeds': 200,
}



def align_image(
    src_im:np.ndarray, 
    ref_im:np.ndarray, 
    crop_list=None,
    use_autocorr:bool=True, 
    precision_fold:int=100, 
    min_good_drifts:int=3, 
    drift_diff_th:float=1.,
    all_channels:list=default_channels, 
    ref_all_channels:list=None, 
    fiducial_channel=default_fiducial_channel,
    correction_args={},
    fitting_args={},
    match_distance_th=2.,
    verbose=True, 
    detailed_verbose=False,                      
    ):
    """Function to align one image by either FFT or spot_finding"""
    
    #from ..io_tools.load import correct_fov_image
    #from ..spot_tools.fitting import fit_fov_image
    #from ..spot_tools.fitting import select_sparse_centers
    from skimage.registration import phase_cross_correlation
    #print("**", type(src_im), type(ref_im))
    ## check inputs
    # correciton keywords
    _correction_args = {_k:_v for _k,_v in _default_align_corr_args.items()}
    _correction_args.update(correction_args)
    # fitting keywords
    _fitting_args = {_k:_v for _k,_v in _default_align_fitting_args.items()}
    _fitting_args.update(fitting_args)
    
    # check crop_list:
    if crop_list is None:
        crop_list = generate_drift_crops(_correction_args['single_im_size'])
    for _crop in crop_list:
        if np.shape(np.array(_crop)) != (3,2):
            raise IndexError(f"crop should be 3x2 np.ndarray.")
    # check channels
    _all_channels = [str(_ch) for _ch in all_channels]
    # check bead_channel
    _fiducial_channel = str(fiducial_channel)
    if _fiducial_channel not in all_channels:
        raise ValueError(f"bead channel {_fiducial_channel} not exist in all channels given:{_all_channels}")
    # check ref_all_channels
    if ref_all_channels is None:
        _ref_all_channels = _all_channels
    else:
        _ref_all_channels = [str(_ch) for _ch in ref_all_channels]
    
    ## process source image
    # define result flag
    _result_flag = -1
    # process image
    if isinstance(src_im, np.ndarray):
        if verbose:
            print(f"-- start aligning given source image to", end=' ')
        _src_im = src_im
    elif isinstance(src_im, str):
        if verbose:
            print(f"-- start aligning file {src_im}.", end=' ')
        if not os.path.isfile(src_im) or src_im.split('.')[-1] != 'dax':
            raise IOError(f"input src_im: {src_im} should be a .dax file!")
        _src_im = load_image_base(
            src_im, [_fiducial_channel], 
            verbose=detailed_verbose,
            )[0][0]
    else:
        raise IOError(f"Wrong input file type, {type(src_im)} should be .dax file or np.ndarray")
    
    ## process reference image
    if isinstance(ref_im, np.ndarray):
        if verbose:
            print(f"given reference image.")
        _ref_im = ref_im
    elif isinstance(ref_im, str):
        if verbose:
            print(f"reference file:{ref_im}.")
        if not os.path.isfile(ref_im) or ref_im.split('.')[-1] != 'dax':
            raise IOError(f"input ref_im: {ref_im} should be a .dax file!")
        _ref_im = load_image_base(
            ref_im, [_fiducial_channel], 
            verbose=detailed_verbose,
            )[0][0]
    else:
        raise IOError(f"Wrong input ref file type, {type(ref_im)} should be .dax file or np.ndarray")

    if np.shape(_src_im) != np.shape(_ref_im):
        raise IndexError(f"shape of target image:{np.shape(_src_im)} and reference image:{np.shape(_ref_im)} doesnt match!")

    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = tuple([slice(*np.array(_c,dtype=np.int32)) for _c in _crop])
        _crop_src_ims.append(_src_im[_s])
        _crop_ref_ims.append(_ref_im[_s])
    ## align two images
    _drifts = []
    for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
        _start_time = time.time()
        if use_autocorr:
            if detailed_verbose:
                print("--- use auto correlation to calculate drift.")
            # calculate drift with autocorr
            _dft, _error, _phasediff = phase_cross_correlation(_rim, _sim, 
                                                               upsample_factor=precision_fold)
        else:
            if detailed_verbose:
                print("--- use beads fitting to calculate drift.")
            # source
            _src_spots = fit_fov_image(_sim, _fiducial_channel, 
                verbose=detailed_verbose,
                **_fitting_args) # fit source spots
            _sp_src_cts = select_sparse_centers(_src_spots[:,1:4], match_distance_th) # select sparse source spots
            # reference
            _ref_spots = fit_fov_image(_rim, _fiducial_channel, 
                verbose=detailed_verbose,
                **_fitting_args)
            _sp_ref_cts = select_sparse_centers(_ref_spots[:,1:4], match_distance_th, 
                                                verbose=detailed_verbose) # select sparse ref spots
            #print(_sp_src_cts, _sp_ref_cts)
            
            # align
            _dft, _paired_src_cts, _paired_ref_cts = align_beads(
                _sp_src_cts, _sp_ref_cts,
                _sim, _rim,
                use_fft=True,
                match_distance_th=match_distance_th, 
                return_paired_cts=True,
                verbose=detailed_verbose,
            )
            _dft = _dft * -1 # beads center is the opposite as cross correlation
        # append 
        _drifts.append(_dft) 
        if verbose:
            print(f"-- drift {_i}: {np.around(_dft, 2)} in {time.time()-_start_time:.3f}s.")

        # detect variance within existing drifts
        _mean_dft = np.nanmean(_drifts, axis=0)
        if len(_drifts) >= min_good_drifts:
            _dists = np.linalg.norm(_drifts-_mean_dft, axis=1)
            _kept_drift_inds = np.where(_dists <= drift_diff_th)[0]
            if len(_kept_drift_inds) >= min_good_drifts:
                _updated_mean_dft = np.nanmean(np.array(_drifts)[_kept_drift_inds], axis=0)
                _result_flag = 1
                if verbose:
                    print(f"--- drifts for crops:{_kept_drift_inds} pass the thresold, exit cycle.")
                break
    
    if '_updated_mean_dft' not in locals():
        if verbose:
            print(f"-- return a sub-optimal drift")
        _drifts = np.array(_drifts)
        # select top 3 drifts
        from scipy.spatial.distance import pdist, squareform
        _dist_mat = squareform(pdist(_drifts))
        np.fill_diagonal(_dist_mat, np.inf)
        # select closest pair
        _sel_inds = np.array(np.unravel_index(np.argmin(_dist_mat), np.shape(_dist_mat)))
        _sel_drifts = list(_drifts[_sel_inds])
        # select closest 3rd drift
        _sel_drifts.append(_drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))])
        if detailed_verbose:
            print(f"--- select drifts: {np.round(_sel_drifts, 2)}")
        # return mean
        _updated_mean_dft = np.nanmean(_sel_drifts, axis=0)
        _result_flag = 0

    return  _updated_mean_dft, _result_flag