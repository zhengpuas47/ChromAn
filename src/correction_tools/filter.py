
import numpy as np
import time, sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from default_parameters import default_correction_folder, default_ref_channel

def gaussian_deconvolution(im, gfilt_size=2, niter=1):
    """Gaussian blurred image divided by image itself"""
    from scipy.ndimage import gaussian_filter
    decon_im = im.copy().astype(np.float32)
    for _iter in np.arange(niter):
        decon_im = decon_im / gaussian_filter(decon_im, gfilt_size)
    
    return decon_im

def gaussian_high_pass_filter(image, sigma=4, truncate=2.5):
    """Apply gaussian high pass filter to given image"""
    from scipy.ndimage import gaussian_filter
    lowpass = gaussian_filter(np.array(image, dtype=np.float32), sigma, mode='nearest', truncate=truncate)
    gauss_highpass = image.astype(np.float32) - lowpass
    gauss_highpass[lowpass > image] = 0
    return gauss_highpass.astype(image.dtype)

def gaussian_highpass_correction(
    ims,
    channels,
    correction_pf=None,
    correction_folder=default_correction_folder,
    ref_channel=default_ref_channel,
    sigma:float=4, 
    truncate:float=2.5,
    rescale=True,
    verbose=True,
):
    """Correct gaussian highpass for images from same dax file"""
    _total_gaussian_highpass_start = time.time()
    if verbose:
        print(f"- Start gaussian_highpass correction for channels:{channels}.")
    image_size = np.array(ims[0].shape)
    if len(ims) != len(channels):
        raise IndexError(f"length of gaussian_highpass images and channels doesn't match, exit.")
    # No need for correction_pf
    _correction_pf = correction_pf
    _correction_folder = correction_folder
    _ref_channel = ref_channel
    # apply corrections
    _corrected_ims = []
    for _im, _ch in zip(ims, channels):
        _gaussian_highpass_time = time.time()
        _cim = gaussian_high_pass_filter(
            _im, 
            sigma=sigma,
            truncate=truncate,
        )
        # rescale
        if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
            _min,_max = 0, np.iinfo(_im.dtype).max
            _cim = (_cim - np.min(_cim)) / (np.max(_cim) - np.min(_cim)) * _max + _min
            _cim = _cim.astype(_im.dtype)
        else:
            _cim = np.clip(_cim,
                           a_min=np.iinfo(_im.dtype).min, 
                           a_max=np.iinfo(_im.dtype).max).astype(_im.dtype)
        # append
        _corrected_ims.append(_cim.copy())
        del(_cim)
        if verbose:
            print(f"-- corrected gaussian_highpass for channel {_ch} in {time.time()-_gaussian_highpass_time:.3f}s.")
    if verbose:
        print(f"- Finished gaussian_highpass correction in {time.time()-_total_gaussian_highpass_start:.3f}s.")

    # return 
    return _corrected_ims


# remove hot pixels
def Remove_Hot_Pixels(im, dtype=np.uint16, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest', verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)


def hot_pixel_correction(
    ims,
    channels,
    correction_pf=None,
    correction_folder=default_correction_folder,
    ref_channel=default_ref_channel,
    hot_pixel_th:float=0.5, 
    hot_pixel_num_th:float=4,
    interpolation_style='nearest', 
    rescale=True,
    verbose=True,
):
    """Batch correct hot pixels"""
    _total_hot_pixel_start = time.time()
    if verbose:
        print(f"- Start hot_pixel correction for channels:{channels}.")
    image_size = np.array(ims[0].shape)
    if len(ims) != len(channels):
        raise IndexError(f"length of hot_pixel images and channels doesn't match, exit.")
    # No need for correction_pf
    _correction_pf = correction_pf
    _correction_folder = correction_folder
    _ref_channel = ref_channel
    # apply corrections
    _corrected_ims = []
    for _im, _ch in zip(ims, channels):
        _hot_pixel_time = time.time()
        _cim = Remove_Hot_Pixels(
            _im, 
            dtype=_im.dtype,
            hot_pix_th=hot_pixel_th,
            hot_th=hot_pixel_num_th,
            interpolation_style=interpolation_style,
            verbose=verbose,
        )
        # append
        _corrected_ims.append(_cim.copy())
        del(_cim)
        if verbose:
            print(f"-- corrected hot_pixel for channel {_ch} in {time.time()-_hot_pixel_time:.3f}s.")
    if verbose:
        print(f"- Finished hot_pixel correction in {time.time()-_total_hot_pixel_start:.3f}s.")

    # return 
    return _corrected_ims

