# import functions from packages
import os, sys
import time 
import pickle 
import numpy as np
import scipy
import matplotlib.pyplot as plt 
import multiprocessing as mp 

# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# import local variables
from default_parameters import default_correction_folder, default_ref_channel, default_im_size

# import local functions
from file_io.dax_process import DaxProcesser, load_image_base
from file_io.image_crop import crop_neighboring_area
from .chromatic import generate_polynomial_data

# default parameters for bleedthrough profiles
_bleedthrough_default_channels=['750', '647', '561']

_bleedthrough_default_seeding_args={
    'th_seed':1000,
    'max_num_seeds':300,
    'use_dynamic_th':True,
}
_bleedthrough_default_fitting_args={
    "init_w":1.5,
}


def check_bleedthrough_info(
    _info, 
    _rsq_th=0.81, _intensity_th=150., 
    _check_center_position=True, _center_radius=1.):
    """Function to check one bleedthrough pair"""
    if _info['rsquare'] < _rsq_th:
        return False
    elif 'spot' in _info and _info['spot'][0] < _intensity_th:
        return False
    elif _check_center_position:
        _max_inds = np.array(np.unravel_index(np.argmax(_info['ref_im']), np.shape(_info['ref_im'])))
        _im_ct_inds = (np.array(np.shape(_info['ref_im']))-1)/2
        if (_max_inds < _im_ct_inds-_center_radius).all() or (_max_inds > _im_ct_inds+_center_radius).all():
            return False

    return True

def find_bleedthrough_pairs(filename, channel,
                            corr_channels=_bleedthrough_default_channels,
                            seeding_args=dict(),
                            fitting_args=dict(), 
                            intensity_th=1.,
                            crop_size=9, rsq_th=0.81, 
                            check_center_position=True,
                            save_temp=True, save_name=None,
                            overwrite=True, verbose=True,
                            ):
    """Function to generate bleedthrough spot pairs"""
    if 'LinearRegression' not in locals():
        from sklearn.linear_model import LinearRegression
    ## check inputs
    _channel = str(channel)
    if _channel not in corr_channels:
        raise ValueError(f"{_channel} should be within {corr_channels}")
    # init info_dict
    _info_dict = {}
    _load_flags = []
    for _ch in corr_channels:
        if _ch != _channel:
            _basename = os.path.basename(filename).replace('.dax', f'_ref_{_channel}_to_{_ch}.pkl')
            _basename = 'bleedthrough_'+_basename
            _temp_filename = os.path.join(os.path.dirname(filename), _basename)
            if os.path.isfile(_temp_filename) and not overwrite:
                _infos = pickle.load(open(_temp_filename, 'rb'))
                _kept_infos = [_info for _info in _infos 
                    if check_bleedthrough_info(_info, _rsq_th=rsq_th, 
                                               _intensity_th=intensity_th, 
                                               _check_center_position=check_center_position)]

                _info_dict[f"{_channel}_to_{_ch}"] = _kept_infos
                _load_flags.append(1)
            else:
                _info_dict[f"{_channel}_to_{_ch}"] = []
                _load_flags.append(0)
    
    if np.mean(_load_flags) == 1:
        if verbose:
            print(f"-- directly load from saved tempfile in folder: {os.path.dirname(filename)}")
        return _info_dict
        
    ## 1. load this image
    _daxp = DaxProcesser(filename)
    _daxp._load_image(sel_channels=corr_channels)
    ## 2. fit centers for target channel
    _ref_im = getattr(_daxp, f"im_{_channel}")
    _daxp._fit_3D_spots(fit_channels=[_channel], 
                        channel_2_seeding_kwargs={_channel:seeding_args},
                        channel_2_fitting_kwargs={_channel:fitting_args},
                        )
    # threshold intensities
    #_ref_spots = _ref_spots[_ref_spots[:,0] >= intensity_th]
    ## crop
    for _ch in corr_channels:
        if _ch != _channel:
            _im = getattr(_daxp, f"im_{_ch}")
            _key = f"{_channel}_to_{_ch}"
            if verbose:
                print(f"--- finding matched bleedthrough pairs for {_key}")
            # loop through centers to crop
            for _spot in getattr(_daxp, f"spots_{_channel}"):
                _rim = crop_neighboring_area(_ref_im, _spot[1:4], crop_sizes=crop_size)
                _cim = crop_neighboring_area(_im, _spot[1:4], crop_sizes=crop_size)
                # calculate r-square
                _x = np.ravel(_rim)[:,np.newaxis]
                _y = np.ravel(_cim)
                _reg = LinearRegression().fit(_x,_y)        
                _rsq = _reg.score(_x,_y)
                #print(_reg.coef_, _rsq)
                _info = {
                    'coord': _spot[1:4],
                    'spot': _spot,
                    'ref_im': _rim,
                    'bleed_im': _cim,
                    'rsquare': _rsq,
                    'slope': _reg.coef_[0],
                    'intercept': _reg.intercept_,
                    'file':filename,
                }
                _info_dict[_key].append(_info)

    if save_temp:
        for _ch in corr_channels:
            if _ch != _channel:
                _key = f"{_channel}_to_{_ch}"
                _basename = os.path.basename(filename).replace('.dax', f'_ref_{_channel}_to_{_ch}.pkl')
                _basename = 'bleedthrough_'+_basename
                _temp_filename = os.path.join(os.path.dirname(filename), _basename)

                if verbose:
                    print(f"--- saving {len(_info_dict[_key])} points to file:{_temp_filename}")
                pickle.dump(_info_dict[_key], open(_temp_filename, 'wb'))
            else:
                if verbose:
                    print(f"-- channel {_ch} doesn't match {_channel}, skip saving.")

    # only return the information with rsquare large enough
    _kept_info_dict = {_key:[] for _key in _info_dict}
    for _key, _infos in _info_dict.items():
        _kept_infos = [_info for _info in _infos 
            if check_bleedthrough_info(_info, _rsq_th=rsq_th, 
                                        _intensity_th=intensity_th, 
                                        _check_center_position=check_center_position)]
        _kept_info_dict[_key] = _kept_infos

    return _kept_info_dict


def check_bleedthrough_pairs(info_list, outlier_sigma=2, keep_per_th=0.95, max_iter=20, 
                             verbose=True,):
    """Function to check bleedthrough pairs"""
    from scipy.spatial import Delaunay
    # prepare inputs
    if verbose:
        print(f"- check {len(info_list)} bleedthrough pairs.")
    _coords = np.array([_info['coord'] for _info in info_list])
    _slopes = np.array([_info['slope'] for _info in info_list])
    _intercepts = np.array([_info['intercept'] for _info in info_list])
    
    if verbose:
        print(f"- start iteration with outlier_sigma={outlier_sigma:.2f}, keep_percentage={keep_per_th:.2f}")
    _n_iter = 0
    _kept_flags = np.ones(len(_coords), dtype=bool)
    _flags = []
    while(len(_flags) == 0 or np.mean(_flags) < keep_per_th):
        _n_iter += 1
        _start_time = time.time()
        _flags = []
        _tri = Delaunay(_coords[_kept_flags])
        for _i, (_coord, _slope, _intercept) in enumerate(zip(_coords[_kept_flags], 
                                                              _slopes[_kept_flags], 
                                                              _intercepts[_kept_flags])):
            # get neighboring center ids
            _nb_ids = np.array([_simplex for _simplex in _tri.simplices.copy()
                                if _i in _simplex], dtype=np.int32)
            _nb_ids = np.unique(_nb_ids)
            # remove itself
            _nb_ids = _nb_ids[(_nb_ids != _i) & (_nb_ids != -1)]
            #print(_nb_ids)
            # get neighbor slopes
            _nb_coords = _coords[_nb_ids]
            _nb_slopes = _slopes[_nb_ids]
            _nb_intercepts = _intercepts[_nb_ids]
            _nb_weights = 1 / np.linalg.norm(_nb_coords-_coord, axis=1)
            _nb_weights = _nb_weights / np.sum(_nb_weights)
            #print(_nb_slopes)
            # calculate expected slope and compare
            _exp_slope = np.dot(_nb_slopes.T, _nb_weights) 
            _keep_slope = (np.abs(_exp_slope - _slope) <= outlier_sigma * np.std(_nb_slopes))
            # calculate expected intercept and compare
            _exp_intercept = np.dot(_nb_intercepts.T, _nb_weights) 
            _keep_intercept = (np.abs(_exp_intercept - _intercept) <= outlier_sigma * np.std(_nb_intercepts))
            # append keep flags
            _flags.append((_keep_slope and _keep_intercept))

        # update _kept_flags
        _updating_inds = np.where(_kept_flags)[0]
        _kept_flags[_updating_inds] = np.array(_flags, dtype=bool)
        if verbose:
            print(f"-- iter: {_n_iter}, kept in this round: {np.mean(_flags):.3f}, total: {np.mean(_kept_flags):.3f} in {time.time()-_start_time:.3f}s")
        if _n_iter > max_iter:
            if verbose:
                print(f"-- exceed maximum number of iterations, exit.")
            break
    # selected infos
    kept_info_list = [_info for _info, _flag in zip(info_list, _kept_flags) if _flag]
    if verbose:
        print(f"- {len(kept_info_list)} pairs passed.")
    return kept_info_list            



def interploate_bleedthrough_correction_from_channel(
    info_dicts, ref_channel, target_channel, 
    check_info=True, check_params={},
    max_num_spots=1000, min_num_spots=100, 
    single_im_size=default_im_size, ref_center=None,
    fitting_order=2, allow_intercept=True,
    save_temp=True, save_folder=None, 
    make_plots=True, save_plots=True,
    overwrite=False, verbose=True,
    ):
    """Function to interpolate and generate the bleedthrough correction profiles between two channels
    
    """
    _key = f"{ref_channel}_to_{target_channel}"
    # extract info list of correct channels
    _info_list = []
    for _dict in info_dicts:
        if _key in _dict:
            _info_list += list(_dict[_key])
    if len(_info_list) < min_num_spots:
        if verbose:
            print(f"-- not enough spots ({len(_info_list)}) from {ref_channel} to {target_channel}")
        return np.zeros(single_im_size), np.zeros(single_im_size)
    # keep the spot pairs with the highest rsquares
    if len(_info_list) > max_num_spots:
        if verbose:
            print(f"-- only keep the top {max_num_spots} spots from {len(_info_list)} for bleedthrough interpolation.")
    if len(_info_list) > int(max_num_spots):
        _rsquares = np.array([_info['rsquare'] for _info in _info_list])
        _rsq_th = np.sort(_rsquares)[-int(max_num_spots)]
        _info_list = [_info for _info in _info_list if _info['rsquare']>= _rsq_th]
        
    # check
    if check_info:
        _info_list = check_bleedthrough_pairs(_info_list, **check_params)
    # extract information
    _coords = []
    _slopes = []
    _intercepts = []
    for _info in _info_list:
        _coords.append(_info['coord'])
        _slopes.append(_info['slope'])
        _intercepts.append(_info['intercept'])
    
    if len(_coords) < min_num_spots:
        if verbose:
            print(f"-- not enough spots f({len(_coords)}) from {ref_channel} to {target_channel}")
        return np.zeros(single_im_size), np.zeros(single_im_size)
    else:
        if verbose:
            print(f"-- {len(_coords)} spots are used to generate profiles from {ref_channel} to {target_channel}")
        _coords = np.array(_coords)
        _slopes = np.array(_slopes)
        _intercepts = np.array(_intercepts)

        # adjust ref_coords with ref center
        if ref_center is None:
            _ref_center = np.array(single_im_size)[:np.shape(_coords)[1]] / 2
        else:
            _ref_center = np.array(ref_center)[:np.shape(_coords)[1]]
        _ref_coords = _coords - _ref_center[np.newaxis, :]

        # generate_polynomial_data
        _X = generate_polynomial_data(_coords, fitting_order)
        # do the least-square optimization for slope
        _C_slope, _r,_r2,_r3 = scipy.linalg.lstsq(_X, _slopes)   
        _rsq_slope =  1 - np.sum((_X.dot(_C_slope) - _slopes)**2)\
                    / np.sum((_slopes-np.mean(_slopes))**2) # r2 = 1 - SSR/SST   
        print(_C_slope, _rsq_slope)
        
        # do the least-square optimization for intercept
        _C_intercept, _r,_r2,_r3 = scipy.linalg.lstsq(_X, _intercepts)   
        _rsq_intercept =  1 - np.sum((_X.dot(_C_intercept) - _intercepts)**2)\
                    / np.sum((_intercepts-np.mean(_intercepts))**2) # r2 = 1 - SSR/SST   
        print(_C_intercept, _rsq_intercept)
        
        ## generate profiles
        _pixel_coords = np.indices(single_im_size)
        _pixel_coords = _pixel_coords.reshape(np.shape(_pixel_coords)[0], -1)
        _pixel_coords = _pixel_coords - _ref_center[:, np.newaxis]
        # generate predictive pixel coordinates
        _pX = generate_polynomial_data(_pixel_coords.transpose(), fitting_order)
        _p_slope = np.dot(_pX, _C_slope).reshape(single_im_size)
        _p_intercept = np.dot(_pX, _C_intercept).reshape(single_im_size)
        ## save temp if necessary
        
        if save_temp:
            if save_folder is not None and os.path.isdir(save_folder):
                if verbose:
                    print(f"-- saving bleedthrough temp profile from channel: {ref_channel} to channel: {target_channel}.")
                
            else:
                print(f"-- save_folder is not given or not valid, skip.")
            
        ## make plots if applicable
        if make_plots:
            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_p_slope.mean(0))
            plt.colorbar()
            plt.title(f"{ref_channel} to {target_channel}, slope, rsq={_rsq_slope:.3f}")
            if save_plots and (save_folder is not None and os.path.isdir(save_folder)):
                plt.savefig(os.path.join(save_folder, f'bleedthrough_profile_{ref_channel}_to_{target_channel}_slope.png'),
                            transparent=True)
            plt.show()

            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_p_intercept.mean(0))
            plt.colorbar()
            plt.title(f"{ref_channel} to {target_channel}, intercept, rsq={_rsq_intercept:.3f}")
            if save_plots and (save_folder is not None and os.path.isdir(save_folder)):
                plt.savefig(os.path.join(save_folder, f'bleedthrough_profile_{ref_channel}_to_{target_channel}_intercept.png'),
                            transparent=True)
            plt.show()

        return _p_slope, _p_intercept


# Final function to be called to generate bleedthrough correction
def Generate_bleedthrough_correction(bleed_folders, 
                                     corr_channels=_bleedthrough_default_channels, 
                                     parallel=True, num_threads=12, 
                                     start_fov=1, num_images=40,
                                     correction_folder=default_correction_folder, 
                                     seeding_args={},
                                     fitting_args={}, 
                                     intensity_th=150.,
                                     crop_size=9, rsq_th=0.81, check_center=True,
                                     fitting_order=2, generate_2d=True,
                                     interpolate_args={},
                                     make_plots=True, save_plots=True, 
                                     save_folder=None, 
                                     save_name='bleedthrough_correction',
                                     overwrite_temp=False, overwrite_profile=False,
                                     verbose=True,
                                     
                                     ):
    """Function to generate bleedthrough profiles """
    ## 0. inputs
    _seeding_args = {_k:_v for _k,_v in _bleedthrough_default_seeding_args.items()}
    _seeding_args.update(seeding_args) # update with input info
    _fitting_args = {_k:_v for _k,_v in _bleedthrough_default_fitting_args.items()}
    _fitting_args.update(fitting_args) # update with input info
    ## 1. savefiles
    if save_folder is None:
        save_folder = bleed_folders[0]
    filename_base = save_name
    
    ## 2. select_fov_names
    fov_names = [_fl for _fl in os.listdir(bleed_folders[0]) 
                if _fl.split('.')[-1]=='dax']
    sel_fov_names = [_fl for _fl in sorted(fov_names, key=lambda v:int(v.split('.dax')[0].split('_')[-1]))]
    sel_fov_names = sel_fov_names[int(start_fov):int(start_fov)+int(num_images)]
    # load test_image:
    _test_images, all_channels = load_image_base(os.path.join(bleed_folders[0], fov_names[0]), verbose=verbose)
    single_im_size = np.array(_test_images[0].shape)
    ## 2. get correction args
    _correction_args = {
        'correction_folder': correction_folder,
        'single_im_size': single_im_size,
        'all_channels': all_channels,
    }
    # add channel info
    for _ch in corr_channels:
        filename_base += f"_{_ch}"
    # add dimension info
    if generate_2d:
        for _d in _correction_args['single_im_size'][-2:]:
            filename_base += f'_{int(_d)}'
    else:
        for _d in _correction_args['single_im_size']:
            filename_base += f'_{int(_d)}'
    saved_profile_filename = os.path.join(save_folder, filename_base+'.npy')
    # check existance
    if os.path.isfile(saved_profile_filename) and not overwrite_profile:
        if verbose:
            print(f"+ bleedthrough correction profiles already exists. direct load the profile")
        _bleed_profiles = np.load(saved_profile_filename, allow_pickle=True)

    
    ### not exist: start processing
    else:
        if verbose:
            print(f"+ generating bleedthrough profiles.")

        ## 3. prepare args to generate info
        # assemble args
        _bleed_args = []
        for _fov in sel_fov_names:
            for _folder, _ch in zip(bleed_folders, corr_channels):
                _bleed_args.append(
                    (
                        os.path.join(_folder, _fov),
                        _ch,
                        corr_channels,
                        _seeding_args,
                        _fitting_args,
                        intensity_th,
                        crop_size,
                        rsq_th,
                        check_center,
                        True, None,
                        overwrite_temp, verbose,
                    )
                )
                
        ## 4. multi-processing
        if parallel:
            with mp.Pool(num_threads) as _ca_pool:
                if verbose:
                    print(f"++ generating bleedthrough info for {len(sel_fov_names)} images with {num_threads} threads in", end=' ')
                    _multi_start = time.time()
                _info_dicts = _ca_pool.starmap(find_bleedthrough_pairs, 
                                            _bleed_args, chunksize=1)
                
                _ca_pool.close()
                _ca_pool.join()
                _ca_pool.terminate()
            if verbose:
                print(f"{time.time()-_multi_start:.3f}s.")    
        else:
            if verbose:
                print(f"++ generating bleedthrough info for {len(sel_fov_names)} images in", end=' ')
                _multi_start = time.time()
            _info_dicts = [find_bleedthrough_pairs(*_arg) for _arg in _bleed_args]
            if verbose:
                print(f"{time.time()-_multi_start:.3f}s.")    
        ## 5. generate_the_whole_profile
        profile_shape = np.concatenate([np.array([len(corr_channels),
                                                 len(corr_channels)]),
                                       _correction_args['single_im_size']])
        
        bld_corr_profile = np.zeros(profile_shape)
        # loop through two images
        for _ref_i, _ref_ch in enumerate(corr_channels):
            for _tar_i, _tar_ch in enumerate(corr_channels):
                if _ref_ch == _tar_ch:
                    bld_corr_profile[_tar_i, _ref_i] = np.ones(_correction_args['single_im_size'])
                else:
                    _slope_pf, _intercept_pf = interploate_bleedthrough_correction_from_channel(
                        _info_dicts, _ref_ch, _tar_ch, 
                        single_im_size=_correction_args['single_im_size'],
                        fitting_order=fitting_order, 
                        save_folder=save_folder,
                        verbose=True, **interpolate_args,
                    )
                    # dont save the profile if slope is always larger than 1
                    if np.mean(_slope_pf) > 1.:
                        continue                        
                    bld_corr_profile[_tar_i, _ref_i] = _slope_pf

        # compress to 2d
        if generate_2d:
            bld_corr_profile = bld_corr_profile.mean(2)
        # calculate inverse matrix for each pixel
        if verbose:
            print(f"-- generating inverse matrix.")
        _bleed_profiles = np.zeros(np.shape(bld_corr_profile), dtype=np.float32)
        for _i in range(np.shape(bld_corr_profile)[-2]):
            for _j in range(np.shape(bld_corr_profile)[-1]):
                if generate_2d:
                    _bleed_profiles[:,:,_i,_j] = np.linalg.inv(bld_corr_profile[:,:,_i,_j])
                else:
                    for _z in range(np.shape(bld_corr_profile)[-3]):
                        _bleed_profiles[:,:,_z,_i,_j] = np.linalg.inv(bld_corr_profile[:,:,_z,_i,_j])
        
        # reshape the profile
        ## 5. save
        if verbose:
            print(f"-- saving to file:{saved_profile_filename}")
        np.save(saved_profile_filename, _bleed_profiles.reshape(np.concatenate([[len(corr_channels)**2],
                                                                   np.shape(_bleed_profiles)[2:]])
                                                    ))
        
    return _bleed_profiles

def bleedthrough_correction(
    ims, 
    channels,
    correction_pf=None,
    correction_folder=default_correction_folder,
    ref_channel=None,
    rescale=True,
    verbose=True,
):
    """Function to correct matching bleedthrough
    ims and channels should match one to one.
    """
    _total_bleedthrough_start = time.time()
    if verbose:
        print(f"- Start bleedthrough correction for channels:{channels}.")
    image_size = np.array(ims[0].shape)
    if len(ims) != len(channels):
        raise IndexError(f"length of bleedthrough images and channels doesn't match, exit.")
    # load correction_pf if not given
    if correction_pf is None:
        from .load_corrections import load_correction_profile
        correction_pf = load_correction_profile(
            'bleedthrough', 
            channels,
            correction_folder=correction_folder,
            ref_channel=ref_channel,
            all_channels=channels,
            im_size=image_size,
            verbose=verbose,
        )
    # apply correction
    _corrected_ims = []
    for _ich, _ch1 in enumerate(channels):
        # new image is the sum of all intensity contribution from images in corr_channels
        _bleedthrough_start = time.time()
        _dtype = ims[_ich].dtype
        _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
        # init image
        _cim = np.zeros(image_size)
        for _jch, _ch2 in enumerate(channels):
            _cim += ims[_jch] * correction_pf[_ich, _jch]
        # rescale
        if rescale: # (np.max(_cim) > _max or np.min(_cim) < _min)
            _cim = (_cim - np.min(_cim)) / (np.max(_cim) - np.min(_cim)) * _max + _min
            _cim = _cim.astype(_dtype)
        _cim = np.clip(_cim, a_min=_min, a_max=_max).astype(_dtype)
        _corrected_ims.append(_cim.copy())
        # release RAM
        del(_cim)
        # print time
        if verbose:
            print(f"-- corrected bleedthrough for channel {_ch1} in {time.time()-_bleedthrough_start:.3f}s.")
    if verbose:
        print(f"- finish bleedthrough correction in {time.time()-_total_bleedthrough_start:.3f}s. ")

    return _corrected_ims