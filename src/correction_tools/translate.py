import numpy as np
import time, os, sys
from scipy.ndimage import map_coordinates

# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from default_parameters import default_correction_folder, default_ref_channel

def warp_3D_images(
    ims, 
    corr_channels, 
    corr_drift=True,
    drift=None, 
    drift_channels=None,
    corr_chromatic=True,
    correction_pf=None,
    chromatic_channels=None,
    correction_folder=default_correction_folder,
    ref_channel=default_ref_channel,
    warp_order=3,
    border_mode='grid-constant',
    rescale=True,
    verbose=True,
    ):
    """Function to warp 3D images from the same dax file"""
    from scipy.ndimage import map_coordinates
    _total_warp_start = time.time()
    if verbose:
        print(f"- Start 3D warpping for channels:{corr_channels}.")
    image_size = np.array(ims[0].shape)
    if len(ims) != len(corr_channels):
        raise IndexError(f"length of warp images and channels doesn't match, exit.")
    # check inputs
    if corr_drift:
        if drift is None:
            raise ValueError(f"drift not given to warp image. ")
        _drift = np.array(drift)
        if drift_channels is None:
            drift_channels = corr_channels
        else:
            drift_channels = list(drift_channels)
    if corr_chromatic:
        if chromatic_channels is None:
            chromatic_channels = corr_channels
        else:
            chromatic_channels = list(chromatic_channels)
        # only load channels that do chromatic correction
        if correction_pf is None:
            from .load_corrections import load_correction_profile
            correction_pf = load_correction_profile(
                'chromatic', 
                chromatic_channels,
                correction_folder=correction_folder,
                ref_channel=ref_channel,
                all_channels=list(corr_channels) + [ref_channel],
                im_size=image_size,
                verbose=verbose,
            )
    # apply corrections
    _corrected_ims = []
    for _im, _ch in zip(ims, corr_channels):
        _warp_time = time.time()
        # 1. get coordiates to be mapped
        single_im_size = np.array(_im.shape)
        _coords = np.meshgrid( np.arange(single_im_size[0]), 
                np.arange(single_im_size[1]), 
                np.arange(single_im_size[2]), )
        _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary
        # 2. calculate corrected coordinates if chormatic abbrev.
        if corr_chromatic and _ch in chromatic_channels and correction_pf[_ch] is not None:
            _coords = _coords + correction_pf[_ch]
        # 3. apply drift if necessary
        if corr_drift and _ch in drift_channels and _drift.any():
            _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
        # 4. map coordinates
        _cim = map_coordinates(
            _im, 
            _coords.reshape(_coords.shape[0], -1),
            order=warp_order,
            mode=border_mode, cval=np.min(_im)
            )
        _cim = np.clip(_cim,
                       a_min=np.iinfo(_im.dtype).min, 
                       a_max=np.iinfo(_im.dtype).max).astype(_im.dtype)
        _cim = _cim.reshape(np.shape(_im)).astype(_im.dtype)
        # append
        _corrected_ims.append(_cim.copy())
        # clear
        del(_cim)
        del(_coords)
        if verbose:
            print(f"-- corrected warp for channel {_ch} in {time.time()-_warp_time:.3f}s.")
    if verbose:
        print(f"- Finished warp correction in {time.time()-_total_warp_start:.3f}s.")

    # return
    return _corrected_ims


def old_warp_3d_image(image, drift, chromatic_profile=None, 
                  warp_order=1, border_mode='constant', 
                  verbose=False):
    """Warp image given chromatic profile and drift"""
    _start_time = time.time()
    # 1. get coordiates to be mapped
    single_im_size = np.array(image.shape)
    _coords = np.meshgrid( np.arange(single_im_size[0]), 
            np.arange(single_im_size[1]), 
            np.arange(single_im_size[2]), )
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary 
    # 2. calculate corrected coordinates if chormatic abbrev.
    if chromatic_profile is not None:
        _coords = _coords + chromatic_profile 
    # 3. apply drift if necessary
    _drift = np.array(drift)
    if _drift.any():
        _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(image, 
                               _coords.reshape(_coords.shape[0], -1),
                               order=warp_order,
                               mode=border_mode, cval=np.min(image))
    _corr_im = _corr_im.reshape(np.shape(image)).astype(image.dtype)
    if verbose:
        print(f"-- finish warp image in {time.time()-_start_time:.3f}s. ")
    return _corr_im


