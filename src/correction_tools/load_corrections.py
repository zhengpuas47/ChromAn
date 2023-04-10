import os, pickle
import numpy as np
from ..default_parameters import *


def load_correction_profile(corr_type, 
                            corr_channels=default_corr_channels, 
                            correction_folder=default_correction_folder, 
                            all_channels=default_channels,
                            ref_channel=default_ref_channel, 
                            im_size=default_im_size, 
                            verbose=False):
    """Function to load chromatic/illumination correction profile
    Inputs:
        corr_type: type of corrections to be loaded
        corr_channels: all correction channels to be loaded
    Outputs:
        _pf: correction profile, np.ndarray for bleedthrough, dict for illumination/chromatic
    """
    ## check inputs
    # type
    _allowed_types = ['chromatic', 'illumination', 'bleedthrough', 'chromatic_constants']
    _type = str(corr_type).lower()
    if _type not in _allowed_types:
        raise ValueError(f"Wrong input corr_type, should be one of {_allowed_types}")
    # channel
    _all_channels = [str(_ch) for _ch in all_channels]
    _corr_channels = [str(_ch) for _ch in corr_channels]
    for _channel in _corr_channels:
        if _channel not in _all_channels:
            raise ValueError(f"Wrong input channel:{_channel}, should be one of {_all_channels}")
    _ref_channel = str(ref_channel).lower()
    if _ref_channel not in _all_channels:
        raise ValueError(f"Wrong input ref_channel:{_ref_channel}, should be one of {_all_channels}")

    ## start loading file
    if verbose:
        print(f"-- loading {_type} correction profile from file", end=':')
    if _type == 'bleedthrough':
        _basename = _type+'_correction' \
            + '_' + '_'.join(sorted(_corr_channels, key=lambda v:-int(v))) \
            + '_' + str(im_size[-2])+'_'+str(im_size[-1])+'.npy'
        if verbose:
            print(_basename)
        _pf = np.load(os.path.join(correction_folder, _basename), allow_pickle=True)
        _pf = _pf.reshape(len(_corr_channels), len(_corr_channels), im_size[-2], im_size[-1])
    elif _type == 'chromatic':
        if verbose:
            print('')
        _pf = {}
        for _channel in _corr_channels:
            if _channel != _ref_channel:
                _basename = _type+'_correction' \
                + '_' + str(_channel) + '_' + str(_ref_channel)
                for _d in im_size:
                    _basename += f'_{int(_d)}'
                _basename += '.npy'

                if verbose:
                    print('\t',_channel,_basename)
                _pf[_channel] = np.load(os.path.join(correction_folder, _basename), allow_pickle=True)
            else:
                if verbose:
                    print('\t',_channel, None)
                _pf[_channel] = None
    elif _type == 'chromatic_constants':
        if verbose:
            print('')
        _pf = {}
        for _channel in _corr_channels:
            if _channel != _ref_channel:
                _basename = _type.split('_')[0]+'_correction' \
                + '_' + str(_channel) + '_' + str(_ref_channel) 
                for _d in im_size:
                    _basename += f'_{int(_d)}'
                _basename += '_const.pkl'
                
                if verbose:
                    print('\t',_channel,_basename)
                _pf[_channel] = pickle.load(open(os.path.join(correction_folder, _basename), 'rb') )
            else:
                if verbose:
                    print('\t',_channel, None)
                _pf[_channel] = None
    elif _type == 'illumination':
        if verbose:
            print('')
        _pf = {}
        for _channel in _corr_channels:
            _basename = _type+'_correction' \
            + '_' + str(_channel) \
            + '_' + str(im_size[-2])+'x'+str(im_size[-1])+'.npy'
            if verbose:
                print('\t',_channel,_basename)
            _pf[_channel] = np.load(os.path.join(correction_folder, _basename), allow_pickle=True)

    return _pf 