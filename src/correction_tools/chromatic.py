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

# import local functions
from spot_tools.match_spots import find_paired_centers
from file_io.image_crop import crop_neighboring_area
from default_parameters import default_correction_folder, default_fiducial_channel
from file_io.dax_process import DaxProcesser
# required chromatic parameters
_chromatic_ref_channel='647'

_chromatic_default_drift_args={
}
_chromatic_default_seeding_args={
    'th_seed':500,
    'max_num_seeds':300,
    'use_dynamic_th':True,
}
_chromatic_default_fitting_args={
    "init_w":1.5,
}


def generate_chromatic_function(chromatic_const_file, drift=None):
    """Function to generate a chromatic abbrevation translation function from
    _const.pkl file"""

    if isinstance(chromatic_const_file, dict):
        _info_dict = {_k:_v for _k,_v in chromatic_const_file.items()}
    elif isinstance(chromatic_const_file, str):
        _info_dict = pickle.load(open(chromatic_const_file, 'rb'))
    elif chromatic_const_file is None:
        if drift is None:
            #print('empty_function')
            def _shift_function(_coords, _drift=drift): 
                return _coords
            return _shift_function
        else:
            _info_dict ={
                'constants': [np.array([0]) for _dft in drift],
                'fitting_orders': np.zeros(len(drift),dtype=np.int32),
                'ref_center': np.zeros(len(drift)),
            }
    else:
        raise TypeError(f"Wrong input chromatic_const_file")

    # extract info
    _consts = _info_dict['constants']
    _fitting_orders = _info_dict['fitting_orders']
    _ref_center = _info_dict['ref_center']
    # drift
    if drift is None:
        _drift = np.zeros(len(_ref_center))
    else:
        _drift = drift[:len(_ref_center)]
    
    def _shift_function(_coords, _drift=_drift, 
                        _consts=_consts, 
                        _fitting_orders=_fitting_orders, 
                        _ref_center=_ref_center,
                        ):
        """generated translation function with constants and drift"""
        # return empty if thats the case
        if len(_coords) == 0:
            return _coords
        else:
            _coords = np.array(_coords)

        if np.shape(_coords)[1] == len(_ref_center):
            _new_coords = np.array(_coords).copy()
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _new_coords = np.array(_coords).copy()[:,1:1+len(_ref_center)]
        else:
            raise ValueError(f"Wrong input coords")

        _shifts = []
        for _i, (_const, _order) in enumerate(zip(_consts, _fitting_orders)):
            # calculate dX
            _X = generate_polynomial_data(_new_coords- _ref_center[np.newaxis,:], 
                                          _order)
            # calculate dY
            _dy = np.dot(_X, _const)
            _shifts.append(_dy)
        _shifts = np.array(_shifts).transpose()

        # generate corrected coordinates
        _corr_coords = _new_coords - _shifts + _drift

        # return as input
        if np.shape(_coords)[1] == len(_ref_center):
            _output_coords = _corr_coords
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _output_coords = np.array(_coords).copy()
            _output_coords[:,1:1+len(_ref_center)] = _corr_coords
        return _output_coords
    
    # return function
    return _shift_function


# basic function to geenrate chromatic abbrevation profiles and constants
def Generate_chromatic_abbrevation(chromatic_folder, ref_folder, 
                                   chromatic_channel, 
                                   ref_channel=_chromatic_ref_channel,
                                   fiducial_channel=default_fiducial_channel,
                                   parallel=True, num_threads=12, 
                                   start_fov=0, num_images=48,
                                   correction_folder = default_correction_folder,
                                   drift_args={},
                                   seeding_args={},
                                   fitting_args={},
                                   matching_args={},
                                   crop_size=9, rsq_th=0.9,
                                   fitting_orders=1, ref_center=None,
                                   make_plots=True, save_plots=True, 
                                   save_folder=None, 
                                   save_name='chromatic_correction',
                                   overwrite_temp=False, overwrite_profile=False,
                                   verbose=True,
                                   ):
    """Generate chromatic abbrevation profile from fitted pair of spots"""
    from file_io.dax_process import load_image_base
    ## 0. inputs
    ## 1. select matched fovs
    fov_names = [_fl for _fl in os.listdir(chromatic_folder) 
                if _fl.split('.')[-1]=='dax']
    ref_fov_names = [_fl for _fl in os.listdir(ref_folder) 
                    if _fl.split('.')[-1]=='dax']
    sel_fov_names = [_fl for _fl in sorted(fov_names, key=lambda v:int(v.split('.dax')[0].split('_')[-1])) if _fl in ref_fov_names]
    sel_fov_names = sel_fov_names[int(start_fov):int(start_fov)+int(num_images)]
    # load test_image:
    _test_images, all_channels = load_image_base(os.path.join(chromatic_folder, fov_names[0]), verbose=verbose)
    single_im_size = np.array(_test_images[0].shape)
    ## 2. get correction args
    _correction_args = {
        'correction_folder': correction_folder,
        'single_im_size': single_im_size,
        'all_channels': all_channels,
    }
    # update illumination correction profile
    try:
        from .load_corrections import load_correction_profile
        _correction_args['illumination_profile'] = \
            load_correction_profile('illumination', 
                                    corr_channels=[str(ref_channel), 
                                                   str(chromatic_channel), 
                                                   str(fiducial_channel)], 
                                    correction_folder=correction_folder, 
                                    all_channels=all_channels,
                                    ref_channel=ref_channel, 
                                    im_size=single_im_size, 
                                    verbose=verbose)
    except:
        pass
    _drift_args = {_k:_v for _k,_v in _chromatic_default_drift_args.items()}
    _drift_args.update(drift_args) # update with input info
    _seeding_args = {_k:_v for _k,_v in _chromatic_default_seeding_args.items()}
    _seeding_args.update(seeding_args) # update with input info
    _fitting_args = {_k:_v for _k,_v in _chromatic_default_fitting_args.items()}
    _fitting_args.update(fitting_args) # update with input info
    
    ## 1. savefiles
    if save_folder is None:
        save_folder = chromatic_folder

    filename_base = save_name + '_' + str(chromatic_channel)+'_'+str(ref_channel)
    # add dimension info
    for _d in _correction_args['single_im_size']:
        filename_base += f'_{int(_d)}'
    # full name
    saved_profile_filename = os.path.join(save_folder, filename_base+'.npy')
    saved_const_filename = os.path.join(save_folder, filename_base+'_const.pkl')
    # check existence
    if os.path.isfile(saved_profile_filename) and os.path.isfile(saved_const_filename) and not overwrite_profile:
        if verbose:
            print(f"+ chromatic abbrevation profiles already exists. direct load profiles")
        _ca_profiles = np.load(saved_profile_filename, allow_pickle=True)
        _const_infos = np.load(saved_const_filename, allow_pickle=True)
        _ca_constants = _const_infos['constants']
        _ca_rsqs = _const_infos['rsquares']    

    else:
        ## 3. prepare args to generate info

        # assemble args
        _chromatic_args = [(
            os.path.join(chromatic_folder, _fov),
            os.path.join(ref_folder, _fov),
            chromatic_channel, 
            ref_channel,
            fiducial_channel,
            _correction_args,
            _drift_args,
            _seeding_args,
            _fitting_args,
            matching_args,
            crop_size,
            rsq_th,
            True, None,
            overwrite_temp, verbose) for _fov in sel_fov_names]

        ## 4. multi-processing
        if parallel:
            with mp.Pool(num_threads) as _ca_pool:
                if verbose:
                    print(f"++ generating chromatic info for {len(_chromatic_args)} images in {num_threads} threads in", end=' ')
                    _multi_start = time.time()
                spot_infos = _ca_pool.starmap(find_chromatic_spot_pairs, 
                                            _chromatic_args, chunksize=1)
                
                _ca_pool.close()
                _ca_pool.join()
                _ca_pool.terminate()
            if verbose:
                print(f"{time.time()-_multi_start:.3f}s.")    
        else:
            if verbose:
                print(f"++ generating chromatic info for {len(_chromatic_args)} images in", end=' ')
                _multi_start = time.time()
            spot_infos = [find_chromatic_spot_pairs(*_arg) for _arg in _chromatic_args]
            if verbose:
                print(f"{time.time()-_multi_start:.3f}s.")    
        ## 5. summarize spots from multiple fovs
        _shift_dists = []
        _ref_coords = []
        for _infos in spot_infos:
            for _info in _infos:
                _shift_dist = _info['ca_coord'] - _info['ref_coord'] # +_info['drift'] # removed drift because warpped images
                _ref_coord = (_info['ref_coord'] + _info['ca_coord']) /2
                # append
                _shift_dists.append(_shift_dist)
                _ref_coords.append(_ref_coord)
                
        _shift_dists = np.array(_shift_dists)
        _ref_coords = np.array(_ref_coords)
        # adjust ref_coords with ref center
        if ref_center is None:
            _ref_center = _correction_args['single_im_size'][:np.shape(_ref_coords)[1]] / 2
        else:
            _ref_center = np.array(ref_center)[:np.shape(_ref_coords)[1]]
        _ref_coords = _ref_coords - _ref_center[np.newaxis, :]

        ## 6. do ploynomial fitting
        # fitting order
        _dim = np.shape(_shift_dists)[1]
        if isinstance(fitting_orders, int) or isinstance(fitting_orders, np.int32):
            _fitting_orders = np.ones(_dim, dtype=np.int32)*int(fitting_orders)
        elif isinstance(fitting_orders, list) or isinstance(fitting_orders, np.ndarray):
            _fitting_orders = np.array(fitting_orders)[:_dim]
        else:
            raise TypeError(f"Wrong input type for fitting_orders")
        if verbose:
            print(f"++ fitting polynomial orders: {_fitting_orders}")
        # initialize profiles and constants
        _ca_profiles = []
        _ca_constants = []
        _ca_rsqs = []
        for _i, _max_order in enumerate(_fitting_orders):
            # generate columns for polynomial fitting
            _X = generate_polynomial_data(_ref_coords, _max_order)
            # generate column for result
            _y = _shift_dists[:, _i]
            
            # do the least-square optimization
            _C,_r,_r2,_r3 = scipy.linalg.lstsq(_X, _y)   
            _rsquare =  1 - np.sum((_X.dot(_C) - _y)**2)\
                        / np.sum((_y-np.mean(_y))**2) # r2 = 1 - SSR/SST
            if verbose:
                print(f"-- constants: {_C} with rsquare={_rsquare}")
            # save constants
            _ca_constants.append(_C)
            _ca_rsqs.append(_rsquare)
            # generate profiles
            _pixel_coords = np.indices(_correction_args['single_im_size'])
            _pixel_coords = _pixel_coords.reshape(np.shape(_pixel_coords)[0], -1)
            _pixel_coords = _pixel_coords - _ref_center[:, np.newaxis]
            # generate predictive pixel coordinates
            _pX = generate_polynomial_data(_pixel_coords.transpose(), _max_order)
            # calculate profile
            _py = np.dot(_pX, _C).reshape(_correction_args['single_im_size'])
            _ca_profiles.append(_py)

        ## 7. save profiles and constants
        if verbose:
            print(f"++ saving new profiles into folder: {save_folder}")
            # save profile
            np.save(saved_profile_filename.replace('.npy',''), _ca_profiles)
            # constant file 
            _const_dict = {
                'fitting_orders': _fitting_orders,
                'constants': _ca_constants,
                'rsquares': _ca_rsqs,
                'ref_center': _ref_center,
            }
            pickle.dump(_const_dict, open(saved_const_filename, 'wb'))

    ## 8. plots
    if make_plots:
        for _i, (_pf, _rsq) in enumerate(zip(_ca_profiles, _ca_rsqs)):
            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_pf.mean(tuple(np.arange(len(np.shape(_pf))-2))))
            plt.colorbar()
            plt.title(f"shift axis {_i}, rsq={_rsq:.3f}")
            if save_plots:
                plt.savefig(saved_profile_filename.replace('.npy', f'_{_i}.png'),
                            transparent=True)
            plt.show()

    return _ca_profiles, _ca_constants


## function to be called by multiprocessing:
# generate chromatic info for each fov
def find_chromatic_spot_pairs(ca_filename:str,
                              ref_filename:str,
                              ca_channel:str,
                              ref_channel:str='647', fiducial_channel:str='488',
                              correction_args={},
                              drift_args={},
                              seeding_args={},
                              fitting_args={},
                              matching_args={},
                              crop_size=9, rsq_th=0.9,
                              save_temp=True, save_name=None,
                              overwrite=False, verbose=True, 
                              ):
    """Function to generate chromatic abbrevation spot pairs
    Parameters:
        ca_filename: chromatic abbrevation image filename, string
        
    Returns:
    """
    if 'LinearRegression' not in locals():
        from sklearn.linear_model import LinearRegression
        
    # temp_file
    _basename = os.path.basename(ca_filename).replace('.dax', f'_channel_{ca_channel}_ref_{ref_channel}.pkl')
    _basename = 'chromatic_'+_basename
    temp_filename = os.path.join(os.path.dirname(ca_filename), _basename)
    if os.path.isfile(temp_filename) and not overwrite:
        if verbose:
            print(f"-- directly load from temp_file:{temp_filename}")
        _infos = pickle.load(open(temp_filename,'rb'))
    else:
        # load reference image
        _ref_daxp = DaxProcesser(
            ref_filename, CorrectionFolder=correction_args.get("correction_folder", None),
            FiducialChannel=fiducial_channel, RefCorrectionChannel=ref_channel,
            verbose=verbose,
        )
        _ref_daxp._load_image(sel_channels=[str(ref_channel), str(fiducial_channel)])
        _ref_daxp._corr_illumination(
            correction_channels=[str(ref_channel), str(fiducial_channel)],
            correction_pf=correction_args.get("illumination_profile", None),
        )
        # fitting
        _ref_daxp._fit_3D_spots(
            fit_channels=[str(ref_channel)], 
            channel_2_seeding_kwargs={str(ref_channel):seeding_args},
            channel_2_fitting_kwargs={str(ref_channel):fitting_args},
        )
        # load chromatic_abbrevated image
        _daxp = DaxProcesser(
            ca_filename, CorrectionFolder=correction_args.get("correction_folder", None),
            FiducialChannel=fiducial_channel, RefCorrectionChannel=ref_channel,
            verbose=verbose,
        )
        _daxp._load_image(sel_channels=[str(ca_channel), str(fiducial_channel)])
        _daxp._corr_illumination(
            correction_channels=[str(ca_channel), str(fiducial_channel)],
            correction_pf=correction_args.get("illumination_profile", None),
        )
        _daxp._calculate_drift(
            RefImage=getattr(_ref_daxp, f"im_{fiducial_channel}"),
            drift_kwargs=drift_args,
        )
        _daxp._corr_warpping_drift_chromatic(
            correction_channels=[str(ca_channel)],
            corr_drift=True, corr_chromatic=False,
            ref_channel=ref_channel,
        )
        # fitting
        _daxp._fit_3D_spots(
            fit_channels=[str(ca_channel)], 
            channel_2_fitting_kwargs={str(ca_channel):fitting_args},
            channel_2_seeding_kwargs={str(ref_channel):seeding_args},
            channel_2_seeds={str(ca_channel):\
                             getattr(_ref_daxp, f"spots_{ref_channel}")
                            }
        )

        # extract spots
        _ref_spots = getattr(_ref_daxp, f"spots_{ref_channel}")
        _ca_spots = getattr(_daxp, f"spots_{ca_channel}")
        # pair spots
        _new_dft, _ca_cts, _ref_cts = find_paired_centers(_ca_spots, _ref_spots,
                            **matching_args,
                            return_paired_cts=True,
                           )
        
        # loop through each spot, crop
        _infos = []
        for _ca_ct, _ref_ct in zip(_ca_cts, _ref_cts):
            # crop images
            _rim = crop_neighboring_area(getattr(_ref_daxp, f"im_{ref_channel}"), 
                                         _ref_ct, crop_size)
            _cim = crop_neighboring_area(getattr(_daxp, f"im_{ca_channel}"), 
                                         _ca_ct, crop_size)
            # calculate r-square
            _x = np.ravel(_rim)[:,np.newaxis]
            _y = np.ravel(_cim)
            _reg = LinearRegression().fit(_x,_y)        
            _rsq = _reg.score(_x,_y)
            
            if _rsq >= rsq_th:
                _info_dict = {
                    'ref_coord': _ref_ct,
                    'ca_coord': _ca_ct,
                    'drift': _daxp.drift,
                    'ref_im': _rim,
                    'ca_im': _cim,
                    'rsquare': _rsq,
                    'slope': _reg.coef_,
                    'intercept': _reg.intercept_,
                    'ca_file':ca_filename,
                    'ref_file':ref_filename,
                }
                _infos.append(_info_dict)

        if save_temp:
            if verbose:
                print(f"--- saving {len(_infos)} points to file:{temp_filename}")
            pickle.dump(_infos, open(temp_filename, 'wb'))
    
    return _infos


def generate_polynomial_data(coords, max_order):
    """function to generate polynomial data
    Args:
        coords: coordinates, np.ndarray, n_points by n_dimensions
        max_order: maximum order of polynomial, int
    Return:
        _X: data for polynomial, n_points by n_columns
    """
    import itertools
    _X = []
    for _order in range(int(max_order)+1):
        for _lst in itertools.combinations_with_replacement(
                coords.transpose(), _order):
            # initialize one column
            _xi = np.ones(np.shape(coords)[0])
            # calculate product
            for _v in _lst:
                _xi *= _v
            # append
            _X.append(_xi)
    # transpose to n_points by n_columns
    _X = np.array(_X).transpose()
    
    return _X