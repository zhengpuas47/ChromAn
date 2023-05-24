import time, warnings

import numpy as np
from .spot_class import Spots3D

_default_seeding_parameters = {
    'th_seed': 500,
    'max_num_seeds': None,
}
_default_fitting_parameters = {
    'radius_fit': 5, # size of gaussian kernel
    'init_w': 1.5, # size of initial sigma square
}

class SpotFitter(object):

    def __init__(
        self, 
        image, 
        seeding_parameters=dict(),
        fitting_parameters=dict(), 
        verbose=True,
        ):
        """Class of spot fitting"""
        super().__init__()
        self.image = image
        self.seeding_parameters = _default_seeding_parameters
        self.seeding_parameters.update(seeding_parameters)
        self.fitting_parameters = _default_fitting_parameters
        self.fitting_parameters.update(fitting_parameters)
        self.verbose = verbose
        return

    def seeding(
        self, 
        seeding_kwargs=None,
        overwrite=True,
        ):
        """Function to call get_seeds"""
        if not hasattr(self, 'seeds') or overwrite:
            if self.verbose:
                print("- start SpotFitter seeding")
            # kwargs
            _seeding_kwargs = self.seeding_parameters
            if isinstance(seeding_kwargs, dict):
                _seeding_kwargs.update(seeding_kwargs)
            if len(_seeding_kwargs) == 0:
                warnings.warn(f"No fitting kwargs provided, use default parameters")
            # seed
            self.seeds = SpotFitter.get_seeds(
                self.image, verbose=self.verbose,
                **_seeding_kwargs,
            )
        else:
            if self.verbose:
                print("- seeds already exist, skip. ")

    def CPU_fitting(
        self, 
        fitting_kwargs=None,
        remove_boundary_points=False,
        normalization=None, 
        normalization_kwargs=dict(),      
        ):
        """Run fitting"""
        from .bintu_fitting import iter_fit_seed_points
        if not hasattr(self, 'seeds'):
            raise AttributeError("Seeds doesn't exist, pleaase call seeding function first.")
        if self.verbose:
            print(f"-- start fitting spots with {len(self.seeds)} seeds, ", end='')
            _fit_time = time.time()
        ## fitting
        _fitting_kwargs = self.fitting_parameters
        if isinstance(fitting_kwargs, dict):
            _fitting_kwargs.update(fitting_kwargs)
        if len(_fitting_kwargs) == 0:
            warnings.warn(f"No fitting kwargs provided, use default parameters")
        # fit
        _fitter = iter_fit_seed_points(
            self.image, self.seeds.to_coords().T, 
            #init_w=init_sigma, weight_sigma=weight_sigma,
            **_fitting_kwargs,
        )    
        # fit
        _fitter.firstfit()
        # check
        _fitter.repeatfit()
        # get spots
        _spots = Spots3D(np.array(_fitter.ps))
        _spots = _spots[np.sum(np.isnan(_spots),axis=1)==0] # remove NaNs
        # filtering:
        if remove_boundary_points:
            _kept_spots = SpotFitter.remove_edge_points(_spots, )
        else:
            _kept_spots = _spots
        # normalization
        if normalization == 'local':
            from ..file_io.image_crop import generate_neighboring_crop
            _backs = []
            for _pt in _kept_spots:
                _crop = generate_neighboring_crop(
                    _pt.to_coords(),
                    crop_size=_fitter.radius_fit*2,
                    single_im_size=np.array(np.shape(self.image)))
                _cropped_im = self.image[_crop.to_slices()]
                _backs.append(SpotFitter.find_image_background(_cropped_im, **normalization_kwargs))
            if self.verbose:
                print(f"normalize local background for each spot, ", end='')
            _kept_spots[:,0] = _kept_spots[:,0] / np.array(_backs)
        elif normalization == 'global':
            _back = SpotFitter.find_image_background(self.image, **normalization_kwargs)
            if self.verbose:
                print(f"normalize global background:{_back:.2f}, ", end='')
            _kept_spots[:,0] = _kept_spots[:,0] / _back
        # save attribute
        if self.verbose:
            print(f"{len(_spots)} fitted in {time.time()-_fit_time:.3f}s.")
        self.spots = _kept_spots
        return

    @staticmethod
    # integrated function to get seeds
    def get_seeds(
        im, 
        max_num_seeds=None, 
        th_seed=200, 
        th_seed_per=95, 
        use_percentile=False,
        sel_center=None, 
        seed_radius=30,
        gfilt_size=0.75, 
        background_gfilt_size=7.5,
        filt_size=3, 
        min_edge_distance=2,
        use_dynamic_th=True, 
        dynamic_niters=10, 
        min_dynamic_seeds=1,
        remove_hot_pixel=True, 
        hot_pixel_th=3,
        verbose=False,
        )->np.ndarray:
        """Function to fully get seeding pixels given a image and thresholds.
        Inputs:
        im: image given, np.ndarray, 
        num_seeds: number of max seeds number, int default=-1, 
        th_seed: seeding threshold between max_filter - min_filter, float/int, default=150, 
        use_percentile: whether use percentile to determine seed_th, bool, default=False,
        th_seed_per: seeding percentile in intensities, float/int of percentile, default=95, 
        sel_center: selected center coordinate to get seeds, array-like, same dimension as im, 
            default=None (whole image), 
        seed_radius: square frame radius of getting seeds, int, default=30,
        gfilt_size: gaussian filter size for max_filter image, float, default=0.75, 
        background_gfilt_size: gaussian filter size for min_filter image, float, default=10,
        filt_size: filter size for max/min filter, int, default=3, 
        min_edge_distance: minimal allowed distance for seed to image edges, int/float, default=3,
        use_dynamic_th: whetaher use dynamic th_seed, bool, default=True, 
        dynamic_niters: number of iterations used for dynamic th_seed, int, default=10, 
        min_dynamic_seeds: minimal number of seeds to get with dynamic seeding, int, default=1,
        remove_hot_pixel: whether remove pixel that generate multiple seeds, bool, default=True,
        hot_pixel_th: max threshold to remove for seeds within this pixel, int, default=3 
        verbose: whether say something!, bool, default=False,
        """
        from scipy.stats import scoreatpercentile
        from scipy.ndimage import maximum_filter,minimum_filter,gaussian_filter
        # check inputs
        if not isinstance(im, np.ndarray):
            raise TypeError(f"image given should be a numpy.ndarray, but {type(im)} is given.")
        if th_seed_per >= 100 or th_seed_per <= 50:
            use_percentile = False
            print(f"th_seed_per should be a percentile > 50, invalid value given ({th_seed_per}), so not use percentile here.")
        # crop image if sel_center is given
        if sel_center is not None:
            if len(sel_center) != len(np.shape(im)):
                raise IndexError(f"num of dimensions should match for selected center and image given.")
            # get selected center and cropping neighbors
            _center = np.array(sel_center, dtype=np.int64)
            _llims = np.max([np.zeros(len(im.shape)), _center-seed_radius], axis=0)
            _rlims = np.min([np.array(im.shape), _center+seed_radius], axis=0)
            _lims = np.array(np.transpose(np.stack([_llims, _rlims])), dtype=np.int64)
            _lim_crops = tuple([slice(_l,_r) for _l,_r in _lims])
            # crop image
            _im = im[_lim_crops]
            # get local centers for adjustment
            _local_edges = _llims
        else:
            _local_edges = np.zeros(len(np.shape(im)))
            _im = im
        # get threshold
        if use_percentile:
            _th_seed = scoreatpercentile(im, th_seed_per) - scoreatpercentile(im, (100-th_seed_per)/2)
        else:
            _th_seed = th_seed
        if verbose:
            _start_time = time.time()
            if not use_dynamic_th:
                print(f"-- start seeding image with threshold: {_th_seed:.2f}", end='; ')
            else:
                print(f"-- start seeding image, th={_th_seed:.2f}", end='')
        ## do seeding
        if not use_dynamic_th:
            dynamic_niters = 1 # setting only do seeding once
        else:
            dynamic_niters = int(dynamic_niters)
        # front filter:
        if gfilt_size:
            _max_im = np.array(gaussian_filter(_im, gfilt_size), dtype=_im.dtype)
        else:
            _max_im = np.array(_im, dtype=_im.dtype)
        _max_ft = np.array(maximum_filter(_max_im, int(filt_size)) == _max_im, dtype=bool)
        # background filter
        if background_gfilt_size:
            _min_im = np.array(gaussian_filter(_im, background_gfilt_size), dtype=_im.dtype)
        else:
            _min_im = np.array(_im, dtype=_im.dtype)
        _min_ft = np.array(minimum_filter(_min_im, int(filt_size)) != _min_im, dtype=bool)
        # generate map
        _local_maximum_mask = (_max_ft & _min_ft).astype(bool)
        _diff_ft = (_max_im.astype(np.float32) - _min_im.astype(np.float32))
        # clear RAM immediately
        del(_max_im, _min_im)
        del(_max_ft, _min_ft) 
        # iteratively select seeds
        for _iter in range(dynamic_niters):
            # get seed coords
            _current_seed_th = _th_seed * (1-_iter/dynamic_niters)
            #print(_iter, _current_seed_th)
            # should be: local max, not local min, differences large than threshold
            _coords = np.array(np.where(_local_maximum_mask & (_diff_ft >= _current_seed_th))).transpose()
            # remove edges
            if min_edge_distance > 0:
                _coords = SpotFitter.remove_edge_points(_im, _coords, min_edge_distance)
            # if got enough seeds, proceed.
            if len(_coords) >= min_dynamic_seeds:
                break
        # print current th
        if verbose and use_dynamic_th:
            print(f"->{_current_seed_th:.2f}", end=', ')
        # hot pixels
        if remove_hot_pixel:
            _,_x,_y = _coords.transpose()
            _xy_str = [str([np.round(x_,1),np.round(y_,1)]) 
                        for x_,y_ in zip(_x,_y)]
            _unique_xy_str, _cts = np.unique(_xy_str, return_counts=True)
            _keep_hot = np.array([_xy not in _unique_xy_str[_cts>=hot_pixel_th] 
                                for _xy in _xy_str],dtype=bool)
            _coords = tuple(_cs[_keep_hot] for _cs in _coords.transpose())
        # get heights
        _hs = _diff_ft[_coords]
        _final_coords = np.array(_coords) + _local_edges[:, np.newaxis] # adjust to absolute coordinates
        # add intensity
        _final_coords = np.concatenate([_hs[np.newaxis,:], _final_coords])
        # transpose and sort by intensity decreasing order
        _final_coords = np.transpose(_final_coords)[np.flipud(np.argsort(_hs))]
        if verbose:
            print(f"found {len(_final_coords)} seeds in {time.time()-_start_time:.2f}s")
        # truncate with max_num_seeds
        if max_num_seeds is not None and max_num_seeds > 0 and max_num_seeds <= len(_final_coords):
            _final_coords = _final_coords[:np.int64(max_num_seeds)]
            if verbose:
                print(f"--- {max_num_seeds} seeds are kept.")
        return Spots3D(_final_coords)
    
    @staticmethod
    def remove_edge_points(
        image:np.ndarray, 
        points:Spots3D, 
        distance:int=2,
        )->np.ndarray:
        """Remove spots that close to image boundary"""
        _im_size = np.array(image.shape)
        if isinstance(points, Spots3D):
            _coords = points.to_coords()
        else:
            _coords = points[:,-len(_im_size):]
        flags = []
        for _coord in _coords:
            _f = ((_coord >= distance) * (_coord <= _im_size-distance)).all()
            flags.append(_f)
        return points[np.array(flags, dtype=bool)]

    @staticmethod
    def find_image_background(
        im, 
        bin_size=10, 
        make_plot=False, 
        max_iter=10,
        ):
        """Function to calculate image background
        Inputs: 
            im: image, np.ndarray,
            bin_size: size of histogram bin, smaller -> higher precision and longer time,
                float (default: 10)
            make_plot: whether generate plot for background calling, bool (default:False),
            max_iter: maximum number of iteration to perform peak calling, int (default:10),
        Output: 
            _background: determined background level, float
        """
        from scipy.signal import find_peaks
        dtype = im.dtype 
        _cts, _bins = np.histogram(im, 
                                bins=np.arange(np.iinfo(dtype).min, 
                                                np.iinfo(dtype).max,
                                                bin_size)
                                )
        _peaks = []
        # gradually lower height to find at least one peak
        _height = np.size(im)/50
        _iter = 0
        while len(_peaks) == 0:
            _height = _height / 2 
            _peaks, _params = find_peaks(_cts, height=_height)
            _iter += 1
            if _iter > max_iter:
                break
        # select highest peak
        if _iter > max_iter:
            _background = np.nanmedian(im)
        else:
            _sel_peak = _peaks[np.argmax(_params['peak_heights'])]
            # define background as this value
            _background = (_bins[_sel_peak] + _bins[_sel_peak+1]) / 2
        # plot histogram if necessary
        if make_plot:
            import matplotlib.pyplot as plt
            plt.figure(dpi=100)
            plt.hist(np.ravel(im), bins=np.arange(np.iinfo(dtype).min, 
                                                np.iinfo(dtype).max,
                                                bin_size))
            plt.xlim([np.min(im), np.max(im)])     
            plt.show()

        return _background


def multi_fit_image(
        
):
    """Function to seed and fit spots in given field of view"""




# fit the entire field of view image
def fit_fov_image(im, channel, seeds=None, 
                  seed_mask=None,
                  max_num_seeds=500,
                  th_seed=300, th_seed_per=95, use_percentile=False, 
                  use_dynamic_th=True, 
                  dynamic_niters=10, min_dynamic_seeds=1,
                  remove_hot_pixel=True, seeding_kwargs={}, 
                  fit_radius=5, #init_sigma=_sigma_zxy, weight_sigma=0, 
                  normalize_background=False, normalize_local=False, 
                  background_args={}, 
                  fitting_args={}, 
                  remove_boundary_points=True, verbose=True):
    """Function to merge seeding and fitting for the whole fov image"""

    ## check inputs
    th_seed = float(th_seed)
    if verbose:
        print(f"-- start fitting spots in channel:{channel}, ", end='')
        _fit_time = time.time()
    ## seeding
    if seeds is None:
        _seeds = SpotFitter.get_seeds(im, max_num_seeds=max_num_seeds,
                        th_seed=th_seed, th_seed_per=th_seed_per,
                        use_percentile=use_percentile,
                        use_dynamic_th=use_dynamic_th, 
                        dynamic_niters=dynamic_niters,
                        min_dynamic_seeds=min_dynamic_seeds,
                        remove_hot_pixel=remove_hot_pixel,
                        return_h=False, verbose=False,
                        **seeding_kwargs,)
        if verbose:
            print(f"{len(_seeds)} seeded with th={th_seed}, ", end='')
    else:
        _seeds = np.array(seeds)[:,:len(np.shape(im))]
        if verbose:
            print(f"{len(_seeds)} given, ", end='')
    # if no seeds, skip
    if len(_seeds) == 0:
        return np.array([])

    # apply seed mask if given
    if seed_mask is not None:
        _sel_seeds = []
        for _seed in _seeds:
            if seed_mask[tuple(np.round(_seed[:len(np.shape(im))]).astype(np.int64))] > 0:
                _sel_seeds.append(_seed)
        # replace seeds
        _seeds = np.array(_sel_seeds)
        if verbose:
            print(f"{len(_seeds)} selected by mask, ", end='')

    ## fitting
    from .bintu_fitting import iter_fit_seed_points
    _fitter = iter_fit_seed_points(
        im, _seeds.T, radius_fit=fit_radius, 
        #init_w=init_sigma, weight_sigma=weight_sigma,
        **fitting_args,
    )    
    # fit
    _fitter.firstfit()
    # check
    _fitter.repeatfit()
    # get spots
    _spots = np.array(_fitter.ps)
    _spots = _spots[np.sum(np.isnan(_spots),axis=1)==0] # remove NaNs
    # remove all boundary points
    if remove_boundary_points:
        _kept_flags = (_spots[:,1:4] > np.zeros(3)).all(1) \
            * (_spots[:, 1:4] < np.array(np.shape(im))).all(1)
        _spots = _spots[np.where(_kept_flags)[0]]
        pass 
    # normalize intensity if applicable
    if normalize_background and not normalize_local:
        _back = SpotFitter.find_image_background(im, **background_args)
        if verbose:
            print(f"normalize total background:{_back:.2f}, ", end='')
        _spots[:,0] = _spots[:,0] / _back
    elif normalize_local:
        from ..file_io.image_crop import generate_neighboring_crop
        _backs = []
        for _pt in _spots:
            _crop = generate_neighboring_crop(_pt[1:4],
                                              crop_size=fit_radius*2,
                                              single_im_size=np.array(np.shape(im)))
            _cropped_im = im[_crop.to_slices()]
            _backs.append(SpotFitter.find_image_background(_cropped_im, **background_args))
        if verbose:
            print(f"normalize local background for each spot, ", end='')
        _spots[:,0] = _spots[:,0] / np.array(_backs)

    if verbose:
        print(f"{len(_spots)} fitted in {time.time()-_fit_time:.3f}s.")
    return _spots
