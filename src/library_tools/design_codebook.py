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

# Calculation
def HammingDist_for_matrix(matrix):
    from scipy.spatial.distance import pdist, squareform
    _dists = pdist(matrix)**2
    print(f"minimum hamming distance: {np.min(_dists):1n}")
    _distmat = squareform(_dists)
    return _distmat