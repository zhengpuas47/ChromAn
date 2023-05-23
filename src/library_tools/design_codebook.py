import numpy as np

# Conversion
def barcode_to_matrix(barcodes, num_bits=None):
    try:
        _num_bits = int(num_bits)
    except:
        _num_bits = max(len(np.unique(barcodes)), np.max(barcodes)+1)
    _matrix = np.zeros([len(barcodes), _num_bits], dtype=np.int32)
    for _i, _b in enumerate(barcodes):
        _matrix[_i][_b] = 1
    return _matrix

def remove_bad_barcodes(barcodes, min_hamming_dist=4):
    # init kept barcodes:
    _kept_barcodes = barcodes.copy()
    _matrix = barcode_to_matrix(_kept_barcodes)
    _distmap = HammingDist_for_matrix(_matrix)
    _hd = int(np.min(_distmap[np.triu_indices_from(_distmap,1)]))
    while _hd < min_hamming_dist:
        print(_hd, len(_kept_barcodes))
        _bX, _bY = np.where(_distmap < min_hamming_dist)# get bad X and Y
        _X, _Xcount = np.unique(_bX, return_counts=True)
        # find the worst:
        _bad_ind = _X[_Xcount == np.max(_Xcount)][0]
        _kept_barcodes = np.delete(_kept_barcodes, _bad_ind, axis=0)
        # re-calculate hamming distance
        _matrix = barcode_to_matrix(_kept_barcodes)
        _distmap = HammingDist_for_matrix(_matrix)
        _hd = int(np.min(_distmap[np.triu_indices_from(_distmap,1)]))
        #break
    return _kept_barcodes

def select_balanced_subsets(barcodes, sel_num, ):
    """Given all possible barcodes with set Hamming distances and barcodes, """
    # check inputs
    if len(barcodes) < sel_num:
        raise ValueError(f"try to select more barcodes than given. ({sel_num} of {len(barcodes)})")
    print(f"- Selecting {sel_num} barcodes among {len(barcodes)}")
    _barcodes = np.array(barcodes) # internalize
    _dist_mat = HammingDist_for_matrix(barcode_to_matrix(_barcodes)) # pre-calculate distance matrix
    _min_hd = np.unique(_dist_mat)[1]
    #print(f"Minimum Hamming dist: {_min_hd}")
    # init and select
    _sel_inds = [np.random.randint(len(barcodes))] # start with one random
    _bit_usage = np.zeros(len(np.unique(barcodes)))
    _bit_usage[_barcodes[_sel_inds[-1]]] += 1
    while len(_sel_inds) < sel_num:
        #print(_sel_inds)
        # select existing
        _cand_inds = np.setdiff1d(np.arange(len(barcodes)), np.array(_sel_inds))
        _cand_dist_counts = (_dist_mat[np.array(_sel_inds)][:, _cand_inds] == _min_hd).sum(0)
        _cand_usages = np.array([ np.sum(_bit_usage[_barcodes[_i]]) for _i in _cand_inds])
        # for the minimal overlapped bits, select the most balanced one
        _min_overlap_flags = np.where(_cand_dist_counts == np.min(_cand_dist_counts))[0]
        _min_overlap_usages = _cand_usages[_min_overlap_flags]
        _min_usage_flags = np.where(_min_overlap_usages == np.min(_min_overlap_usages))[0]
        _final_cand_inds = _cand_inds[_min_overlap_flags][_min_usage_flags]
        # randomly select among final candidates
        _final_sel_ind = _final_cand_inds[np.random.randint(len(_final_cand_inds))]
        # append
        _sel_inds.append(_final_sel_ind)
        _bit_usage[_barcodes[_sel_inds[-1]]] += 1
        #print(len(_min_overlap_flags), len(_min_usage_flags))
        #print(_sel_inds)
        #break
    # return final barcodes:
    _sel_inds = np.sort(_sel_inds)
    _unused_inds = np.setdiff1d(np.arange(len(barcodes)), np.array(_sel_inds))
    _sel_barcodes = _barcodes[_sel_inds]
    _unused_barcodes = _barcodes[_unused_inds]
    return _sel_barcodes, _unused_barcodes

# Calculation
def HammingDist_for_matrix(matrix):
    from scipy.spatial.distance import pdist, squareform
    _dists = pdist(matrix)**2
    print(f"minimum hamming distance: {np.min(_dists):1n}")
    _distmat = squareform(_dists)
    return _distmat

