import numpy as np
import os, sys
    
def find_paired_centers(tar_cts, ref_cts, drift=None,
                        cutoff=2, dimension=3,  
                        return_paired_cts=True, 
                        return_kept_inds=False,
                        verbose=False):
    """Function to fast find uniquely paired centers given two lists
        of centers and candidate drifts (tar-ref).
    Inputs:
        tar_cts: target centers, 2d-numpy array
        ref_cts: reference centers, 2d-numpy array, 
            numbers don't have to match but dimension should match tar_cts
        drift: rough drift between tar_cts - ref_cts, 1d-numpy array of dim,
        cutoff: unique matching distance cutoff, float (default: 2)
        return_paired_cts: whether return paired center coordinates, bool (default: True)
        verbose: whether say something! bool (default: True)
    Outputs:
        _mean_shift: mean drift calculated from paired centers, 1d numpy array of dim,
    conditional outputs:
        _paired_tar_cts: paired target centers, 2d numpy arrray of n_spots*dim
        _paired_ref_cts: paired reference centers, 2d numpy arrray of n_spots*dim
    """
    from scipy.spatial.distance import cdist
    _dimension = int(dimension)
    _tar_cts = np.array(tar_cts)
    _ref_cts = np.array(ref_cts)
    if np.shape(_tar_cts)[1] > 3:
        _tar_cts = _tar_cts[:,1:1+_dimension]
    if np.shape(_ref_cts)[1] > 3:
        _ref_cts = _ref_cts[:,1:1+_dimension]
        
    if drift is None:
        _drift = np.zeros(np.shape(_tar_cts)[1])
    else:
        _drift = np.array(drift, dtype=np.float64)[:_dimension]
    if verbose:
        print(f"-- aligning {len(_tar_cts)} centers to {len(_ref_cts)} ref_centers, given drift:{np.round(_drift,2)}",
              end=', ')
    # adjust ref centers to match target centers
    _adj_ref_cts = _ref_cts + _drift
    # canculate dists
    _dists = cdist(_tar_cts, _adj_ref_cts)
    _tar_inds, _ref_inds = np.where(_dists <= cutoff)
    # pick only unique ones
    _unique_tar_inds = np.where(np.sum(_dists <= cutoff, axis=1) == 1)[0]
    _unique_ref_inds = np.where(np.sum(_dists <= cutoff, axis=0) == 1)[0]
    # get good pairs
    _unique_pair_inds = [[_t, _r] for _t, _r in zip(_tar_inds, _ref_inds)
                         if _t in _unique_tar_inds \
                         and _r in _unique_ref_inds]
    # acquire paired centers
    _paired_tar_cts = []
    _paired_ref_cts = []
    for _it, _ir in _unique_pair_inds:
        _paired_tar_cts.append(_tar_cts[_it])
        _paired_ref_cts.append(_ref_cts[_ir])
    _paired_tar_cts = np.array(_paired_tar_cts)
    _paired_ref_cts = np.array(_paired_ref_cts)
    
    # calculate mean drift and return
    _new_drift = np.nanmean(_paired_tar_cts - _paired_ref_cts, axis=0)
    if verbose:
        print(f"{len(_paired_tar_cts)} pairs found, updated_drift:{np.round(_new_drift,2)}")

    # return
    _return_args = [_new_drift]
    if return_paired_cts:
        _return_args.append(_paired_tar_cts)
        _return_args.append(_paired_ref_cts)
    if return_kept_inds:
        _paired_tar_inds = np.array(_unique_pair_inds, dtype=np.int)[:,0]
        _paired_ref_inds = np.array(_unique_pair_inds, dtype=np.int)[:,1]
        # append
        _return_args.append(_paired_tar_inds)
        _return_args.append(_paired_ref_inds)
    return tuple(_return_args)

def check_paired_centers(paired_tar_cts, paired_ref_cts, 
                         outlier_sigma=1.5, 
                         return_paired_cts=True,
                         verbose=False):
    """Function to check outlier for paired centers, 
        outlier is judged by whether a drift 
        is significantly different from its neighbors
    Inputs:
        paired_tar_cts: paired target centers, 2d-numpy array, 
        paired_ref_cts: paired reference centers, 2d-numpy array,
            should be exactly same size as paired_tar_cts
        outlier_sigma: cutoff for a drift comparing to neighboring drifts
            (assuming gaussian distribution), float, default: 1.5
        return_paired_cts: whether return paired center coordinates, bool (default: True)
        verbose: whether say something! bool (default: True)
    Outputs:
        _mean_shift: mean drift calculated from paired centers, 1d numpy array of dim,
    conditional outputs:
        _paired_tar_cts: paired target centers, 2d numpy arrray of n_spots*dim
        _paired_ref_cts: paired reference centers, 2d numpy arrray of n_spots*dim
        """
    from scipy.spatial import Delaunay
    _tar_cts = np.array(paired_tar_cts, dtype=np.float)
    _ref_cts = np.array(paired_ref_cts, dtype=np.float)
    _shifts = _tar_cts - _ref_cts
    if verbose:
        print(f"-- check {len(_tar_cts)} pairs of centers", end=', ')
    # initialize new center shifts
    _new_shifts = []
    # use Delaunay to find neighbors for each center pair
    _tri = Delaunay(_ref_cts)
    for _i, (_s, _tc, _rc) in enumerate(zip(_shifts, _tar_cts, _ref_cts)):
        # get neighboring center ids
        _nb_ids = np.array([_simplex for _simplex in _tri.simplices.copy()
                            if _i in _simplex], dtype=np.int)
        _nb_ids = np.unique(_nb_ids)
        # remove itself
        _nb_ids = _nb_ids[(_nb_ids != _i) & (_nb_ids != -1)]
        # get neighbor shifts
        _nb_ref_cts = _ref_cts[_nb_ids]
        _nb_shifts = _shifts[_nb_ids]
        _nb_weights = 1 / np.linalg.norm(_nb_ref_cts-_rc, axis=1)
        # update this shift
        _nb_shift = np.dot(_nb_shifts.T, _nb_weights) / np.sum(_nb_weights)
        _new_shifts.append(_nb_shift)
    _new_shifts = np.array(_new_shifts)
    # select among new_shifts
    _diffs = np.linalg.norm(_new_shifts-_shifts, axis=1)
    _keep_flags = np.array(_diffs < np.mean(_diffs) + np.std(_diffs) * outlier_sigma)
    # cast this keep flags
    _kept_tar_cts = _tar_cts[_keep_flags]
    _kept_ref_cts = _ref_cts[_keep_flags]
    # mean drift
    _new_drift = np.nanmean(_kept_tar_cts-_kept_ref_cts, axis=0)
    if verbose:
        print(f"{len(_kept_tar_cts)} pairs kept. new drift:{np.round(_new_drift,2)}")
    
    # return
    _return_args = [_new_drift]
    if return_paired_cts:
        _return_args.append(_kept_tar_cts)
        _return_args.append(_kept_ref_cts)
    
    return tuple(_return_args)