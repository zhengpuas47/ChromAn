import numpy as np

def barcode_to_matrix(barcodes):
    _num_bits = max(len(np.unique(barcodes)), np.max(barcodes)+1)
    _matrix = np.zeros([len(barcodes), _num_bits], dtype=np.int32)
    for _i, _b in enumerate(barcodes):
        _matrix[_i][_b] = 1
    return _matrix