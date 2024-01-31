import numpy as np
from pandas import DataFrame
import sys, os
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from .file_io import data_organization

# mimic MERLin Optimization, prepare data:
def find_image_background(im, dtype=None, bin_size=10, make_plot=False, max_iter=10):
    """Function to calculate image background
    Inputs: 
        im: image, np.ndarray,
        dtype: data type for image, numpy datatype (default: np.uint16) 
        bin_size: size of histogram bin, smaller -> higher precision and longer time,
            float (default: 10)
    Output: 
        _background: determined background level, float
    """
    from scipy.signal import find_peaks
    if dtype is None:
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

class Optimization_decode():
    """
    An analysis class for decoding MERFISH data
    """
    def __init__(self, codebook_filename, 
                 color_usage_filename=None,
                 parameter_filename=None,
                 verbose=True,
                 ):
        # direct inputs
        self.verbose = verbose
        # load inputs
        self.codebook_filename = codebook_filename
        self._load_codebook()
        self.color_usage_filename = color_usage_filename
        self._load_color_usage()
        # paramters
        self.parameter_filename = parameter_filename
        self.prev_scale_factors = []
        self.prev_backgrounds = []

    def _load_color_usage(self):
        """
        Load color usage from file
        """
        if not hasattr(self, 'color_usage_df') or self.color_usage_df is None:
            if self.verbose:
                print(f"-- loading color usage from file: {self.color_usage_filename}")
            self.color_usage_df = data_organization.Color_Usage(self.color_usage_filename, verbose=False)
        else:
            pass
    
    def _load_codebook(self):
        """
        Load codebook from file
        """
        if not hasattr(self, 'codebook') or self.codebook is None:
            if self.verbose:
                print(f"-- loading codebook from file: {self.codebook_filename}")
            self.codebook = pd.read_csv(self.codebook_filename)
        else:
            pass
    
    def _load_images(self, cropped_ims):
        """
        include loaded images into the class, for now its a test case
        """
        # add loaded images
        self.ims = np.array(cropped_ims)
        return
    
    def _init_scale_factors(
            self, 
            n_bins=4000,      
            scale_max=0.99,
        ) -> np.ndarray:
        """
        initialize scale factors
        """
        self.scale_factors = np.ones((self.ims.shape[0], self.ims.shape[1]), dtype=np.float32)
        data_type = self.ims.dtype
        ## TODO: add reading for bits information
        bits = np.arange(self.ims.shape[1])
        scale_factors, backgrounds = [], []
        for bit in bits:
            # calculate cumulative histogram
            counts, intensities =  np.histogram(np.array(self.ims)[:,bit], bins=np.arange(np.iinfo(data_type).min, 
                                                                                np.iinfo(data_type).max+1,
                                                                                (np.iinfo(data_type).max+1 - np.iinfo(data_type).min)/n_bins))
                                                                                
            intensities = (intensities[:-1] + intensities[1:]) / 2
            cumsum_counts = np.cumsum(counts)
            cumsum_counts = cumsum_counts / cumsum_counts[-1]

            # calculate background
            background = find_image_background(np.array(self.ims)[:,bit], )
            backgrounds.append(background)
            scaling_factor = np.ceil(intensities[np.argmin(np.abs(cumsum_counts - scale_max))])
            #scale_factors.append(max(scaling_factor - background, 1))
            scale_factors.append(scaling_factor+1)
            print(bit, scaling_factor, background) 
        # add attribute
        self.scale_factors = np.array(scale_factors)
        self.backgrounds = np.array(backgrounds)
        return np.array(scale_factors), np.array(backgrounds)
    
    def _decode_foci(
        self, 
        image_series=None, 
        codebook:DataFrame=None,
        scale_factors:np.ndarray=None,
        backgrounds:np.ndarray=None,
        distance_threshold:int=np.sqrt(2), # square root of 2 is the maximum distance allowed as Hamming distance
        magnitute_threshold:int=1,   
        ) -> int:
        """
        Assign barcodes to each foci
        Each image_series has been preprocessed and aligned, 
        then according to the scaling factor and background, 
        this focus is normalized into a vector and assigned to specific barcode based on Euclidean distance
        to one of the existing barcode
        """
        # Inputs
        if image_series is None:
            # num_foci x num_bits x dx x dy x dz
            image_series = self.ims
        if scale_factors is None:
            scale_factors = self.scale_factors
        if backgrounds is None:
            backgrounds = self.backgrounds
        if codebook is None:
            codebook = self.codebook
        _codebook_mat = codebook.iloc[:,1:].values # get codebook mat
        # init assigned_infos:
        self.barcode_ids, self.barcode_dists, self.barcode_mags = [], [], []
        # normalize images
        for _i, _images in enumerate(image_series):
            #_signal_series = scoreatpercentile(_images.reshape((_images.shape[0],-1)), 95, axis=1)
            _signal_series = _images 
            _normalized_series = (_signal_series - backgrounds[:,np.newaxis,np.newaxis,np.newaxis]) / scale_factors[:,np.newaxis,np.newaxis,np.newaxis]        
            # refit, anything not fully normalized forced to be 1 at max
            #_normalized_barcode = np.array([min(scoreatpercentile(_n,99.9),1) for _n in _normalized_series])
            _normalized_barcode = np.array([min(np.max(_n),1) for _n in _normalized_series])
            # calculate Euclidian distance to existing barcodes 
            _barcode_dists = np.linalg.norm(_codebook_mat - _normalized_barcode, axis=1) 
            # check if this could be assigned:
            if np.min(_barcode_dists) < distance_threshold:
                _assigned_barcode_id = np.argmin(_barcode_dists)
                _assigned_dist = np.min(_barcode_dists)

                # calculate magnitute:
                _assigned_mag = np.linalg.norm(_normalized_barcode[_codebook_mat[_assigned_barcode_id]])
                #print(_assigned_barcode_id, _assigned_dist, _assigned_mag)
            else:
                _assigned_barcode_id = -1
                _assigned_dist = np.inf
                _assigned_mag = np.inf
                
            #break
            self.barcode_ids.append(_assigned_barcode_id)
            self.barcode_dists.append(_assigned_dist)
            self.barcode_mags.append(_assigned_mag)

        return self.barcode_ids, self.barcode_dists, self.barcode_mags
    
    # update params
    def _update_background_scale_factors(
        self, 
        codebook:DataFrame=None,
        n_bins=4000,  
        scale_max=0.98,
        ):
        if codebook is None:
            codebook = self.codebook
        _codebook_mat = codebook.iloc[:,1:].values

        if not hasattr(self, 'barcode_ids'):
            raise AttributeError("No barcode_ids detected, probably haven't run any decoding yet.")
        # select successful decoded
        #_sel_barcode_ids = np.array(self.barcode_ids)[np.array(self.barcode_ids) >= 0]
        _sel_barcodes = _codebook_mat[self.barcode_ids]
        ## TODO: add reading for bits information
        bits = np.arange(self.ims.shape[1])
        scale_factors, backgrounds = [], []
        for _bit in bits:
            ## select positive and negative images:
            binary_flags = _sel_barcodes[:,_bit].astype(bool)
            data_type = self.ims.dtype
            pos_ims = np.array(self.ims)[binary_flags & (np.array(self.barcode_ids) >=0), _bit]
            neg_ims = np.array(self.ims)[~binary_flags & (np.array(self.barcode_ids) >=0), _bit]
            ## get scale factor from positive images:
            counts, intensities =  np.histogram(pos_ims, bins=np.arange(np.iinfo(data_type).min, 
                                                                        np.iinfo(data_type).max+1,
                                                                        (np.iinfo(data_type).max+1 - np.iinfo(data_type).min)/n_bins))

            intensities = (intensities[:-1] + intensities[1:]) / 2
            cumsum_counts = np.cumsum(counts)
            cumsum_counts = cumsum_counts / cumsum_counts[-1]
            scaling_factor = np.ceil(intensities[np.argmin(np.abs(cumsum_counts - scale_max))])
            scale_factors.append(scaling_factor+1)
            ## get background from negative
            background = np.mean(neg_ims)#find_image_background(neg_ims, )
            backgrounds.append(background)

            print(_bit, len(pos_ims), len(neg_ims), end=', ')
            print(self.scale_factors[_bit], scale_factors[_bit], end=', ')
            print(self.backgrounds[_bit], backgrounds[_bit])
        # store previous factors
        self.prev_scale_factors.append(np.copy(self.scale_factors))
        self.prev_backgrounds.append(np.copy(self.backgrounds))
        # replace attribute with new factors
        self.scale_factors = np.array(scale_factors)
        self.backgrounds = np.array(backgrounds)

        return scale_factors, backgrounds
    
    # final function to run analysis
    def run_analysis_decode(
        self,
        images,
        n_iter=10,
        params={},
        ):
        """Final call function to run iterations"""
        # load image
        self._load_images(images)
        # init
        self._init_scale_factors()
        for _iter in range(n_iter):
            # decode
            self._decode_foci()
            # update_param
            self._update_background_scale_factors()
            
        return

