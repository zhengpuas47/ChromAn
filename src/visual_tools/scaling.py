import numpy as np

def _rescaling(im, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(im)
    if vmax is None:
        vmax = np.max(im)
    _res_im = np.clip(im, vmin, vmax)
    _res_im = (_res_im - vmin) / (vmax - vmin)
    _res_im = (_res_im * np.iinfo(np.uint8).max ).astype(np.uint8)
    return _res_im

def rescale_by_percentile(im, 
                          min_max_percent=[50,99.95],
                          verbose=True,
                          ):
    from scipy.stats import scoreatpercentile
    vmin, vmax = scoreatpercentile(im, min(min_max_percent)), scoreatpercentile(im, max(min_max_percent))
    if verbose:
        print(vmin, vmax)
    return _rescaling(im, vmin=vmin, vmax=vmax)