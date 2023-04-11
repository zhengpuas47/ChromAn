import os, re, time, h5py, warnings, pickle
import numpy as np 
import xml.etree.ElementTree as ET
# default params
from ..default_parameters import *
# usful functions
from ..correction_tools.load_corrections import load_correction_profile
from ..spot_tools.spot_class import Spots3D

class Reader(object):
    """
    The superclass containing those functions that 
    are common to reading a STORM movie file.

    Subclasses should implement:
     1. __init__(self, filename, verbose = False)
        This function should open the file and extract the
        various key bits of meta-data such as the size in XY
        and the length of the movie.

     2. loadAFrame(self, frame_number)
        Load the requested frame and return it as numpy array.
    """
    def __init__(self, filename, verbose = False):
        super(Reader, self).__init__()
        self.filename = filename
        self.fileptr = None
        self.verbose = verbose

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def averageFrames(self, start = False, end = False):
        """
        Average multiple frames in a movie.
        """
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames 

        length = end - start
        average = np.zeros((self.image_height, self.image_width), np.float32)
        for i in range(length):
            if self.verbose and ((i%10)==0):
                print(" processing frame:", i, " of", self.number_frames)
            average += self.loadAFrame(i + start)
            
        average = average/float(length)
        return average

    def close(self):
        if self.fileptr is not None:
            self.fileptr.close()
            self.fileptr = None
        
    def filmFilename(self):
        """
        Returns the film name.
        """
        return self.filename

    def filmSize(self):
        """
        Returns the film size.
        """
        return [self.image_width, self.image_height, self.number_frames]

    def loadAFrame(self, frame_number):
        assert frame_number >= 0, "Frame_number must be greater than or equal to 0, it is " + str(frame_number)
        assert frame_number < self.number_frames, "Frame number must be less than " + str(self.number_frames)

class DaxReader(Reader):
    # dax specific initialization
    def __init__(self, filename, swap_axis=False, verbose = 0):
        import os,re
        # save the filenames
        self.filename = filename
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"
        # swap_axis
        self.swap_axis = swap_axis

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            self.fileptr = 0
            if verbose:
                print("dax data not found", filename)

    # Create and return a memory map the dax file
    def loadMap(self):
        if os.path.exists(self.filename):
            if self.bigendian:
                self.image_map = np.memmap(self.filename, dtype='>u2', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
            else:
                self.image_map = np.memmap(self.filename, dtype='uint16', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
        return self.image_map

    # load a frame & return it as a np array
    def loadAFrame(self, frame_number):
        if self.fileptr:
            assert frame_number >= 0, "frame_number must be greater than or equal to 0"
            assert frame_number < self.number_frames, "frame number must be less than " + str(self.number_frames)
            self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
            image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
            if self.swap_axis:
                image_data = np.transpose(np.reshape(image_data, [self.image_width, self.image_height]))
            else:
                image_data = np.reshape(image_data, [self.image_width, self.image_height])
            if self.bigendian:
                image_data.byteswap(True)
            return image_data
    # load full movie and retun it as a np array
    def loadAll(self):
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = -1)
        if self.swap_axis:
            image_data = np.swapaxes(np.reshape(image_data, [self.number_frames,self.image_width, self.image_height]),1,2)
        else:
            image_data = np.reshape(image_data, [self.number_frames,self.image_width, self.image_height])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data
    
    def close(self):
        if self.fileptr.closed:
            print(f"file {self.filename} has been closed.")
        else:
            self.fileptr.close()

def load_image_base(
    filename:str,
    sel_channels:list=None,
    all_channels:list=None,
    ImSize:np.ndarray=None, 
    NbufferFrame:int=default_num_buffer_frames,
    NemptyFrame:int=default_num_empty_frames,
    verbose:bool=False,
):
    """Function to simply load imaage, referenced as base"""

    _load_start = time.time()
    # get all channels
    if all_channels is None:
        _channels = DaxProcesser._FindDaxChannels(filename, verbose=verbose)
    else:
        _channels = [str(_ch) for _ch in all_channels]
    # get selected channels
    if sel_channels is None or len(sel_channels) == 0:
        _sel_channels = _channels
    elif isinstance(sel_channels, list):
        _sel_channels = [str(_ch) for _ch in sel_channels]
    elif isinstance(sel_channels, str) or isinstance(sel_channels, int):
        _sel_channels = [str(sel_channels)]
    else:
        raise ValueError(f"Invalid input for sel_channels")
    # choose loading 
    for _ch in sorted(_sel_channels, key=lambda v: _channels.index(v)):
        if _ch not in _channels:
            raise ValueError(f"channel:{_ch} doesn't exist.")
    # get image size
    if ImSize is None:
        _image_size = DaxProcesser._FindImageSize(filename,
            channels=_channels,
            NbufferFrame=NbufferFrame,
            verbose=verbose,
            )
    else:
        _image_size = np.array(ImSize, dtype=np.int32)
    # Load dax file
    _reader = DaxReader(filename, verbose=verbose)
    _raw_im = _reader.loadAll()
    # split by channel
    _ims = split_im_by_channels(
        _raw_im, _sel_channels,
        all_channels=_channels, single_im_size=_image_size,
        num_buffer_frames=NbufferFrame, num_empty_frames=NemptyFrame,
    )
    if verbose:
        print(f"- Loaded images for channels:{_sel_channels} in {time.time()-_load_start:.3f}s.")
    # save attributes
    return _ims, _sel_channels

class DaxProcesser():
    """Major image processing class for 3D image in DNA-MERFISH,
    including two major parts:
        1. image corrections
        2. spot finding
    """
    def __init__(self, 
                 ImageFilename, 
                 CorrectionFolder=None,
                 Channels=None,
                 FiducialChannel=None,
                 DapiChannel=None,
                 verbose=True,
                 ):
        """Initialize DaxProcessing class"""
        if isinstance(ImageFilename, str) \
            and os.path.isfile(ImageFilename)\
            and ImageFilename.split(os.extsep)[-1] == 'dax':
            self.filename = ImageFilename
        elif not isinstance(ImageFilename, str):
            raise TypeError(f"Wrong input type ({type(ImageFilename)}) for ImageFilename.")
        elif ImageFilename.split(os.extsep)[-1] != 'dax':
            raise TypeError(f"Wrong input file extension, should be .dax")
        else:
            raise OSError(f"image file: {ImageFilename} doesn't exist, exit.")
        if verbose:
            print(f"Initialize DaxProcesser for file:{ImageFilename}")
        # other files together with dax
        self.inf_filename = self.filename.replace('.dax', '.inf') # info file
        self.off_filename = self.filename.replace('.dax', '.off') # offset file
        self.power_filename = self.filename.replace('.dax', '.power') # power file
        self.xml_filename = self.filename.replace('.dax', '.xml') # xml file
        # Correction folder
        self.correction_folder = CorrectionFolder
        # verbose
        self.verbose=verbose
        # Channels
        if Channels is None:
            _loaded_channels = DaxProcesser._FindDaxChannels(self.filename, verbose=self.verbose)
            if _loaded_channels is None:
                self.channels = default_channels
            else:
                self.channels = _loaded_channels
        elif isinstance(Channels, list) or isinstance(Channels, np.ndarray):
            self.channels = list(Channels)
        else:
            raise TypeError(f"Wrong input type for Channels")
        if FiducialChannel is not None and str(FiducialChannel) in self.channels:
            setattr(self, 'fiducial_channel', str(FiducialChannel))
        if DapiChannel is not None and str(DapiChannel) in self.channels:
            setattr(self, 'dapi_channel', str(DapiChannel))
        # Log for whether corrections has been done:
        self.correction_log = {_ch:{} for _ch in self.channels}
        
    def _check_existance(self):
        """Check the existance of the full set of Dax file"""
        # return True if all file exists
        return os.path.isfile(self.filename) \
            and os.path.isfile(self.inf_filename) \
            and os.path.isfile(self.off_filename) \
            and os.path.isfile(self.power_filename) \
            and os.path.isfile(self.xml_filename) \

    def _load_image(self, 
                    sel_channels=None,
                    ImSize=None, 
                    NbufferFrame=default_num_buffer_frames,
                    NemptyFrame=default_num_empty_frames,
                    save_attrs=True, overwrite=False,
                    ):
        """Function to load and parse images by channels,
            assuming that for each z-layer, all channels has taken a frame in the same order
        """
        # init loaded channels
        if not hasattr(self, 'loaded_channels'):
            setattr(self, 'loaded_channels', [])
        # get selected channels
        if sel_channels is None:
            _sel_channels = self.channels
        elif isinstance(sel_channels, list):
            _sel_channels = [str(_ch) for _ch in sel_channels]
        elif isinstance(sel_channels, str) or isinstance(sel_channels, int):
            _sel_channels = [str(sel_channels)]
        else:
            raise ValueError(f"Invalid input for sel_channels")
        # choose loading channels
        _loading_channels = []
        for _ch in sorted(_sel_channels, key=lambda v: self.channels.index(v)):
            if hasattr(self, f"im_{_ch}") and not overwrite:
                continue
            else:
                _loading_channels.append(_ch)
        # Load:
        _ims, _ = load_image_base(
            filename=self.filename, 
            sel_channels=_loading_channels, 
            all_channels=self.channels,
            ImSize=ImSize, 
            NbufferFrame=NbufferFrame,
            NemptyFrame=NemptyFrame,
            verbose=self.verbose,
        )
        if not hasattr(self, 'image_size') and len(_ims) > 0:
            self.image_size = np.array(_ims[0].shape)
        # save attributes
        if save_attrs:
            for _ch, _im in zip(_loading_channels, _ims):
                setattr(self, f"im_{_ch}", _im)
            setattr(self, 'num_buffer_frames', NbufferFrame)
            setattr(self, 'num_empty_frames', NemptyFrame)
            self.loaded_channels.extend(_loading_channels)
            # sort loaded
            self.loaded_channels = [_ch for _ch in sorted(self.loaded_channels, key=lambda v: self.channels.index(v))]
            return
        else:
            return _ims, _loading_channels

    def _corr_bleedthrough(self,
                           correction_channels=None,
                           correction_pf=None, 
                           correction_folder=None,
                           rescale=True,
                           save_attrs=True,
                           )->None:
        """Apply bleedthrough correction to remove crosstalks between channels
            by a pre-measured matrix"""
        # find correction channels
        from ..correction_tools.bleedthrough import bleedthrough_correction
        
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
            if str(_ch) != getattr(self, 'fiducial_channel', None) \
                and str(_ch) != getattr(self, 'dapi_channel', None)
            ]
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_bleedthrough', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct bleedthrough already finished, skip. ")
            return 
        ## if not finished, do process
        if correction_folder is None:
            correction_folder = self.correction_folder
        # process
        _corrected_ims = bleedthrough_correction(
            [getattr(self, f"im_{_ch}") for _ch in _correction_channels], 
            _correction_channels,
            correction_pf=correction_pf,
            correction_folder=correction_folder,
            rescale=rescale,
            verbose=self.verbose,
        )
        # after finish, save attr
        if save_attrs:
            for _ch, _im in zip(_correction_channels, _corrected_ims):
                setattr(self, f"im_{_ch}", _im.copy())
            del(_corrected_ims)
            # update log
            for _ch in _correction_channels:
                self.correction_log[_ch]['corr_bleedthrough'] = True
            return 
        else:
            return _corrected_ims, _correction_channels
    # remove hot pixels
    def _corr_hot_pixels_3D(
        self, 
        correction_channels=None,
        hot_pixel_th:float=0.5, 
        hot_pixel_num_th:float=4, 
        save_attrs:bool=True,
        )->None:
        """Remove hot pixel by interpolation"""
        _total_hot_pixel_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        if self.verbose:
            print(f"- Correct hot_pixel for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_hot_pixel', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct hot_pixel already finished, skip. ")
            return 
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_hot_pixel.")
        _corrected_ims = []
        # apply correction
        for _ch in _correction_channels:
            _hot_pixel_time = time.time()
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip hot_pixel correction for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                from ..correction_tools.filter import Remove_Hot_Pixels
                _dtype = _im.dtype
                _im = Remove_Hot_Pixels(_im, _dtype, 
                    hot_pix_th=hot_pixel_th,
                    hot_th=hot_pixel_num_th,
                )
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im.astype(_dtype), )
                    # update log
                    self.correction_log[_ch]['corr_hot_pixel'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"-- corrected hot_pixel for channel {_ch} in {time.time()-_hot_pixel_time:.3f}s.")
        # finish
        if self.verbose:
            print(f"- Finished hot_pixel correction in {time.time()-_total_hot_pixel_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
    # illumination correction                    
    def _corr_illumination(self, 
                           correction_channels=None,
                           correction_pf=None, 
                           correction_folder=None,
                           rescale=True,
                           save_attrs=True,
                           overwrite=False,
                           )->None:
        """Apply illumination correction to flatten field-of-view illumination
            by a pre-measured 2D-array"""
        _total_illumination_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        if self.verbose:
            print(f"- Correct illumination for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_illumination', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct illumination already finished, skip. ")
            return 
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_illumination.")
        # load profile
        if correction_folder is None:
            correction_folder = self.correction_folder
        if correction_pf is None:
            correction_pf = load_correction_profile(
                'illumination', _correction_channels,
                correction_folder=correction_folder,
                ref_channel=_correction_channels[0],
                all_channels=self.channels,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        _corrected_ims = []
        # apply correction
        for _ch in _correction_channels:
            _illumination_time = time.time()
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip illumination correction for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                _dtype = _im.dtype
                _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
                # apply corr
                _im = _im.astype(np.float32) / correction_pf[_ch][np.newaxis,:]
                if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
                    _im = (_im - np.min(_im)) / (np.max(_im) - np.min(_im)) * _max + _min
                _im = np.clip(_im, a_min=_min, a_max=_max)
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im.astype(_dtype), )
                    # update log
                    self.correction_log[_ch]['corr_illumination'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"-- corrected illumination for channel {_ch} in {time.time()-_illumination_time:.3f}s.")
        # finish
        if self.verbose:
            print(f"- Finished illumination correction in {time.time()-_total_illumination_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels

    def _corr_chromatic_functions(self, 
        correction_channels=None,
        correction_pf=None, 
        correction_folder=None,
        ref_channel=default_ref_channel,
        save_attrs=True,
        overwrite=False,
        ):
        """Generate chromatic_abbrevation functions for each channel"""
        from ..correction_tools.chromatic import generate_chromatic_function
        _total_chromatic_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
            if str(_ch) != getattr(self, 'fiducial_channel', None) and str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Generate corr_chromatic_functions for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_chromatic', False) or self.correction_log[_ch].get('corr_chromatic_function', False)
            for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct chromatic function already finished, skip. ")
            return 
        # update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"- Keep channels: {_correction_channels} for corr_chromatic_functions.")
        ## if not finished, do process
        if self.verbose:
            print(f"- Start generating chromatic correction for channels:{_correction_channels}.")
        if correction_folder is None:
            correction_folder = self.correction_folder
        if correction_pf is None:
            correction_pf = load_correction_profile(
                'chromatic_constants', _correction_channels,
                correction_folder=correction_folder,
                all_channels=self.channels,
                ref_channel=ref_channel,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        ## loop through channels to generate functions
        # init corrected_funcs
        _image_size = getattr(self, 'image_size')
        _drift = getattr(self, 'drift', np.zeros(len(_image_size)))
        _corrected_funcs = []
        # apply
        for _ch in _correction_channels:
            if self.verbose:
                _chromatic_time = time.time()
                print(f"-- generate chromatic_shift_function for channel: {_ch}", end=' ')
            _func = generate_chromatic_function(correction_pf[_ch], _drift)
            if save_attrs:
                setattr(self, f"chromatic_func_{_ch}", _func)
            else:
                _corrected_funcs.append(_func)
            if self.verbose:
                print(f"in {time.time()-_chromatic_time:.3f}s")
        if self.verbose:
            print(f"-- finish generating chromatic functions in {time.time()-_total_chromatic_start:.3f}s")
        if save_attrs:
            # update log
            for _ch in _correction_channels:
                self.correction_log[_ch]['corr_chromatic_function'] = True
            return 
        else:
            return _corrected_funcs
        
    def _calculate_drift(
        self, 
        RefImage, 
        FiducialChannel=default_fiducial_channel, 
        precise_align=True,
        use_autocorr=True, 
        drift_kwargs={},
        save_attr=True, 
        save_ref_im=False,
        overwrite=False,
        ):
        """Calculate drift given reference image"""
        if hasattr(self, 'drift') and hasattr(self, 'drift_flag') and not overwrite:
            if self.verbose:
                print(f"- Drift already calculated, skip.")
            return self.drift, self.drift_flag
        # Load drift image
        if FiducialChannel is None and hasattr(self, 'fiducial_channel'):
            FiducialChannel = getattr(self, 'fiducial_channel')
        elif FiducialChannel is not None:
            FiducialChannel = str(FiducialChannel)
        else:
            raise ValueError(f"Wrong input value for FiducialChannel: {FiducialChannel}")
        # get _DriftImage
        if FiducialChannel in self.channels and hasattr(self, f"im_{FiducialChannel}"):
            _DriftImage = getattr(self, f"im_{FiducialChannel}")
        elif FiducialChannel in self.channels and not hasattr(self, f"im_{FiducialChannel}"):
            _DriftImage = self._load_image(sel_channels=[FiducialChannel], 
                                           ImSize=self.image_size,
                                           NbufferFrame=self.num_buffer_frames, NemptyFrame=self.num_empty_frames,
                                           save_attrs=False)[0][0]
            
        else:
            raise AttributeError(f"FiducialChannel:{FiducialChannel} image doesn't exist, exit.")
        if self.verbose:
            print(f"+ Calculate drift with fiducial_channel: {FiducialChannel}")
        if isinstance(RefImage, str) and os.path.isfile(RefImage):
            # if come from the same file, skip
            if RefImage == self.filename:
                if self.verbose:
                    print(f"-- processing ref_image itself, skip.")
                _drift = np.zeros(len(self.image_size))
                _drift_flag = 0
                if save_attr:
                    # drift channel
                    setattr(self, 'fiducial_channel', FiducialChannel)
                    # ref image
                    if save_ref_im:
                        setattr(self, 'ref_im', getattr(self, f"im_{FiducialChannel}"))
                    # drift results
                    setattr(self, 'drift', _drift)
                    setattr(self, 'drift_flag', _drift_flag)
                    return 
                else:
                    return _drift, _drift_flag
            # load RefImage from file and get this image
            _dft_dax_cls = DaxProcesser(RefImage, CorrectionFolder=self.correction_folder,
                                        Channels=None, verbose=self.verbose)
            _dft_dax_cls._load_image(sel_channels=[FiducialChannel], ImSize=self.image_size,
                                     NbufferFrame=self.num_buffer_frames, NemptyFrame=self.num_empty_frames)
            RefImage = getattr(_dft_dax_cls, f"im_{FiducialChannel}")

        elif isinstance(RefImage, np.ndarray) and (np.array(RefImage.shape)==np.array(_DriftImage.shape)).all():
            # directly add
            if save_ref_im:
                setattr(self, 'ref_im', RefImage)
            pass
        else:
            raise ValueError(f"Wrong input of RefImage, should be either a matched sized image, or a filename")
        # align image
        if precise_align:
            from ..correction_tools.alignment import align_image
            _drift, _drift_flag = align_image(
                _DriftImage,
                RefImage, 
                use_autocorr=use_autocorr, fiducial_channel=FiducialChannel,
                verbose=self.verbose, **drift_kwargs,
            )
        else:
            if self.verbose:
                print("-- use auto correlation to calculate rough drift.")
            # calculate drift with autocorr
            from skimage.registration import phase_cross_correlation
            _start_time = time.time()
            _drift, _error, _phasediff = phase_cross_correlation(
                RefImage, _DriftImage, 
                )
            _drift_flag = 2
            if self.verbose:
                print(f"-- calculate rough drift: {_drift} in {time.time()-_start_time:.3f}s. ")
        if save_attr:
            # drift channel
            setattr(self, 'fiducial_channel', FiducialChannel)
            # ref image
            if save_ref_im:
                setattr(self, 'ref_im', RefImage)
            # drift results
            setattr(self, 'drift', _drift)
            setattr(self, 'drift_flag', _drift_flag)
        return _drift, _drift_flag
    # warp image
    def _warp_image(self,
                    drift=None,
                    correction_channels=None,
                    corr_chromatic=True, chromatic_pf=None,
                    correction_folder=None,
                    ref_channel=default_ref_channel,
                    save_attrs=True, overwrite=False,
                    ):
        """Warp image in 3D, this step must give a drift"""
        from scipy.ndimage import map_coordinates
        _total_warp_start = time.time()
        # get drift
        if drift is not None:
            _drift = np.array(drift)
        elif hasattr(self, 'drift'):
            _drift = getattr(self, 'drift')
        else:
            _drift = np.zeros(len(self.image_size))
            warnings.warn(f"drift not given to warp image. ")
        # get channels
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        _chromatic_channels = [_ch for _ch in _correction_channels 
                            if _ch != getattr(self, 'fiducial_channel', None) and _ch != getattr(self, 'dapi_channel', None)]
        # Log
        _ch_2_finish_warp = {_ch: self.correction_log[_ch].get('corr_drift', False) or  not _drift.any() for _ch in _correction_channels}
        _ch_2_finish_chromatic = {_ch: self.correction_log[_ch].get('corr_chromatic', False) for _ch in _chromatic_channels}
        _logs = [_ch_2_finish_warp.get(_ch) and _ch_2_finish_chromatic.get(_ch, True) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Warp drift and chromatic already finished, skip. ")
            return 
        # start warpping
        if self.verbose:
            print(f"- Start warpping images channels:{_correction_channels}.")
        if correction_folder is None:
            correction_folder = self.correction_folder
        # load chromatic warp
        if corr_chromatic and chromatic_pf is None:
            chromatic_pf = load_correction_profile(
                'chromatic', _chromatic_channels,
                correction_folder=correction_folder,
                all_channels=self.channels,
                ref_channel=ref_channel,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        # init corrected_ims
        _corrected_ims = []
        # do warpping
        for _ch in _correction_channels:
            _chromatic_time = time.time()
            # get flag for this channel
            _finish_warp = _ch_2_finish_warp.get(_ch)
            _finish_chromatic = _ch_2_finish_chromatic.get(_ch, True)
            print(_ch, _finish_warp, _finish_chromatic)
            # skip if not required
            if _finish_warp and (_finish_chromatic or not corr_chromatic):
                if self.verbose:
                    print(f"-- skip warpping image for channel {_ch}, no drift or chromatic required.")
                continue
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip warpping image for channel {_ch}, image not detected.")
                continue
            # 1. get coordiates to be mapped
            _coords = np.meshgrid(np.arange(self.image_size[0]), 
                                np.arange(self.image_size[1]), 
                                np.arange(self.image_size[2]), 
                                )
            # transpose is necessary  
            _coords = np.stack(_coords).transpose((0, 2, 1, 3)) 
            _note = f"-- warp image"
            # 2. apply drift if necessary
            if not _finish_warp:
                _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
                _note += f' with drift:{_drift}'
                # update flag
                self.correction_log[_ch]['corr_drift'] = True
            # 3. aaply chromatic if necessary
            if not _finish_chromatic and corr_chromatic:
                _note += ' with chromatic abbrevation' 
                if chromatic_pf[_ch] is None and str(_ch) == ref_channel:
                    pass
                else:                 
                    _coords = _coords + chromatic_pf[_ch]
                # update flag
                self.correction_log[_ch]['corr_chromatic'] = True
            # 4. map coordinates
            if self.verbose:
                print(f"{_note} for channel: {_ch}")
            _im = map_coordinates(_im, 
                                _coords.reshape(_coords.shape[0], -1),
                                mode='nearest').astype(_im.dtype)
            _im = _im.reshape(tuple(self.image_size))

            # 5. save
            if save_attrs:
                setattr(self, f"im_{_ch}", _im,)
            else:
                _corrected_ims.append(_im)
            # release RAM
            del(_im)
            # print time
            if self.verbose:
                print(f"-- finish warpping channel {_ch} in {time.time()-_chromatic_time:.3f}s.")
        
        # print time
        if self.verbose:
            print(f"-- finish warpping in {time.time()-_total_warp_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
    # Gaussian highpass for high-background images
    def _gaussian_highpass(self,                            
                        correction_channels=None,
                        gaussian_sigma=3,
                        gaussian_truncate=2,
                        save_attrs=True,
                        overwrite=False,
                        ):
        """Function to apply gaussian highpass for selected channels"""
        from ..correction_tools.filter import gaussian_high_pass_filter
        _total_highpass_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
                                if str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Apply Gaussian highpass for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_highpass', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"-- Gaussian_highpass for channel:{_correction_channels} already finished, skip. ")
            return 
        # update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for gaussian_highpass.")
        # loop through channels
        _corrected_ims = []
        for _ch in _correction_channels:
            if self.verbose:
                print(f"-- applying gaussian highpass, channel={_ch}, sigma={gaussian_sigma}", end=' ')
                _highpass_time = time.time()
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip gaussian_highpass for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                _dtype = _im.dtype
                _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
                # apply gaussian highpass filter
                _im = gaussian_high_pass_filter(_im, gaussian_sigma, gaussian_truncate)
                _im = np.clip(_im, a_min=_min, a_max=_max).astype(_dtype)
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im)
                    # update log
                    self.correction_log[_ch]['corr_highpass'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"in {time.time()-_highpass_time:.3f}s")
        # finish
        if self.verbose:
            print(f"- Finished gaussian_highpass filtering in {time.time()-_total_highpass_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
            
    # Spot_fitting:
    def _fit_spots(self, fit_channels=None, 
                th_seed=1000, num_spots=None, fitting_kwargs={},
                save_attrs=True, overwrite=False):
        """Fit spots for the entire image"""
        from ..spot_tools.fitting import fit_fov_image
        # total start time
        _total_fit_start = time.time()
        if fit_channels is None:
            fit_channels = self.loaded_channels
        _fit_channels = [str(_ch) for _ch in fit_channels 
                        if str(_ch) != getattr(self, 'fiducial_channel', None) \
                            and str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Fit spots in channels:{_fit_channels}")
        _fit_logs = [hasattr(self, f'spots_{_ch}') and not overwrite for _ch in _fit_channels]
        ## if finished ALL, directly return
        if np.array(_fit_logs).all():
            if self.verbose:
                print(f"-- Fitting for channel:{_fit_channels} already finished, skip. ")
            return 
        # update _fit_channels based on log
        _fit_channels = [_ch for _ch, _log in zip(_fit_channels, _fit_logs) if not _log ]
        # th_seeds
        if isinstance(th_seed, int) or isinstance(th_seed, float):
            _ch_2_thSeed = {_ch:th_seed for _ch in _fit_channels}
        elif isinstance(th_seed, dict):
            _ch_2_thSeed = {str(_ch):_th for _ch,_th in th_seed.items()}
        if self.verbose:
            print(f"-- Keep channels: {_fit_channels} for fitting.")
        _spots_list = []
        for _ch in _fit_channels:
            if self.verbose:
                print(f"-- fitting channel={_ch}", end=' ')
                _fit_time = time.time()
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip fitting for channel {_ch}, image not detected.")
                continue
            # fit
            _th_seed = _ch_2_thSeed.get(_ch, default_seed_th)
            _spots = fit_fov_image(_im, _ch, 
                                th_seed=_th_seed, max_num_seeds=num_spots, 
                                verbose=self.verbose,
                                **fitting_kwargs)
            _cell_ids = np.ones(len(_spots),dtype=np.int32) -1
            if save_attrs:
                setattr(self, f"spots_{_ch}", _spots)
                setattr(self, f"spots_cell_ids_{_ch}", _cell_ids)
            else:
                _spots_list.append(_spots)
        # return
        if self.verbose:
            print(f"-- finish fitting in {time.time()-_total_fit_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _spots_list
    # Spot Fitting 2, by segmentation
    def _fit_spots_by_segmentation(self, channel, seg_label, 
                                   th_seed=500, num_spots=None, fitting_kwargs={},
                                   segment_search_radius=3, 
                                   save_attrs=True, verbose=False):
        """Function to fit spots within each segmentation
        Necessary numbers:
            th_seed: default seeding threshold
            num_spots: number of expected spots (within each segmentation mask)
        """
        from ..segmentation_tools.cells import segmentation_mask_2_bounding_box
        from ..spot_tools.fitting import fit_fov_image
        from .partition_spots import Spots_Partition
        from tqdm import tqdm
        # get drift
        _drift = getattr(self, 'drift', np.zeros(len(self.image_size)))
        # get cell_id
        if self.verbose:
            print(f"- Start fitting spots in each segmentation")
        _cell_ids = np.unique(seg_label)
        _cell_ids = _cell_ids[_cell_ids>0]

        _all_spots, _all_cell_ids = [], []
        for _cell_id in tqdm(_cell_ids):
            _cell_mask = (seg_label==_cell_id)
            _crop = segmentation_mask_2_bounding_box(_cell_mask, 3)

            _local_mask = _cell_mask[_crop.to_slices()]
            _drift_crop = _crop.translate_drift(drift=_drift)
            _drift_local_im = getattr(self, f'im_{channel}')[_drift_crop.to_slices()]
            # fit
            _spots = fit_fov_image(_drift_local_im, str(channel), 
                                th_seed=th_seed, max_num_seeds=num_spots, 
                                verbose=verbose,
                                **fitting_kwargs)
            if len(_spots) > 0:
                # adjust to absolute coordinate per fov
                _spots = Spots3D(_spots)
                _spots[:,_spots.coordinate_indices] = _spots[:,_spots.coordinate_indices] + _drift_crop.array[:,0]
                # keep spots within mask
                _kept_flg = Spots_Partition.spots_to_labels(_cell_mask, _spots, 
                                                            search_radius=segment_search_radius,verbose=False)
                _spots = _spots[_kept_flg>0]
                # append
                if len(_spots) > 0:
                    _all_spots.append(_spots)
                    _all_cell_ids.append(np.ones(len(_spots), dtype=np.int32)*_cell_id)
        # concatenate
        if len(_all_spots) > 0:
            _all_spots = np.concatenate(_all_spots)
            _all_cell_ids = np.concatenate(_all_cell_ids)
        else:
            _all_spots = np.array([])
            _all_cell_ids = np.array([])
            print(f"No spots detected.")
        # save attr
        if save_attrs:
            setattr(self, f"spots_{channel}", _all_spots)
            setattr(self, f"spots_cell_ids_{channel}", _all_cell_ids)
            return
        return _all_spots, _all_cell_ids

    # Saving:
    def _save_to_hdf5(self):
        pass
    def _save_to_npy(self, save_channels, save_folder=None, save_basenames=None):
        if save_folder is None:
            pass

        
    # Loading:
    def _load_from_hdf5(self):
        pass
    @staticmethod
    def _FindDaxChannels(dax_filename,
                         verbose=True,
                         ):
        """Find channels"""
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        try:
            _hal_info = ET.parse(_xml_filename).getroot()
            _shutter_filename = _hal_info.findall('illumination/shutters')[0].text
            _shutter_channels = os.path.basename(_shutter_filename).split(os.extsep)[0].split('_')
            # select all digit names which are channels
            _true_channels = [_ch for _ch in _shutter_channels
                              if len(re.findall(r'^[0-9]+$', _ch))]
            if verbose:
                print(f"-- all used channels: {_true_channels}")
            return _true_channels
        except:
            return None
    @staticmethod
    def _FindGlobalPosition(dax_filename:str,
                            verbose=True) -> np.ndarray:
        """Function to find global coordinates in micron"""
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        try:
            _hal_info = ET.parse(_xml_filename).getroot()
            _position_micron = np.array(_hal_info.findall('acquisition/stage_position')[0].text.split(','), dtype=np.float64)
            return _position_micron
        except:
            raise ValueError(f"Positions not properly parsed")

    @staticmethod
    def _LoadInfFile(inf_filename):
        with open(inf_filename, 'r') as _info_hd:
            _infos = _info_hd.readlines()
        _info_dict = {}
        for _line in _infos:
            _line = _line.rstrip()#.replace(' ','')
            _key, _value = _line.split(' = ')
            _info_dict[_key] = _value
        return _info_dict
    @staticmethod
    def _FindImageSize(dax_filename, 
                       channels=None,
                       NbufferFrame=default_num_buffer_frames,
                       verbose=True,
                       ):
        _inf_filename = dax_filename.replace('.dax', '.inf') # info file
        if channels is None:
            channels = DaxProcesser._FindDaxChannels(dax_filename)                
        try:
            _info_dict = DaxProcesser._LoadInfFile(_inf_filename)
            # get image shape
            _dx,_dy = _info_dict['frame dimensions'].split('x')
            _dx,_dy = int(_dx),int(_dy)
            # get number of frames in z
            _n_frame = int(_info_dict['number of frames'])
            _dz = (_n_frame - 2 * NbufferFrame) / len(channels)
            if _dz == int(_dz):
                _dz = int(_dz)
                _image_size = np.array([_dz,_dx,_dy],dtype=np.int32)
                if verbose:
                    print(f"-- single image size: {_image_size}")
                return _image_size
            else:
                raise ValueError("Wrong num_color, should be integer!")
        except:
            return np.array(default_im_size)
    @staticmethod
    def _LoadSegmentation(segmentation_filename,
                          fov_id=None,
                          verbose=True):
        """Function to load segmentation from file"""
        # check existance
        if not isinstance(segmentation_filename, str) or not os.path.isfile(segmentation_filename):
            raise ValueError(f"invalid segmentation_filename: {segmentation_filename}")
        #load
        if verbose:
            print(f"-- load segmentation from: {segmentation_filename}")
        if segmentation_filename.split(os.extsep)[-1] == 'npy':
            _seg_label = np.load(segmentation_filename)
        elif segmentation_filename.split(os.extsep)[-1] == 'pkl':
            _seg_label = pickle.load(open(segmentation_filename, 'rb'))
        elif segmentation_filename.split(os.extsep)[-1] == 'hdf5' or segmentation_filename.split(os.extsep)[-1] == 'h5':
            with h5py.File(segmentation_filename, 'r') as _f:
                if fov_id is None:
                    fov_id = list(_f.keys())[0]
                _seg_label = _f[str(fov_id)]['dna_mask'][:]
        # return
        return _seg_label

class Writer(object):
    
    def __init__(self, width = None, height = None, **kwds):
        super(Writer, self).__init__(**kwds)
        self.w = width
        self.h = height

    def frameToU16(self, frame):
        frame = frame.copy()
        frame[(frame < 0)] = 0
        frame[(frame > 65535)] = 65535

        return np.round(frame).astype(np.uint16)

        
class DaxWriter(Writer):

    def __init__(self, name, **kwds):
        super(DaxWriter, self).__init__(**kwds)
        
        self.name = name
        if len(os.path.dirname(name)) > 0:
            self.root_name = os.path.dirname(name) + "/" + os.path.splitext(os.path.basename(name))[0]
        else:
            self.root_name = os.path.splitext(os.path.basename(name))[0]
        self.fp = open(self.name, "wb")
        self.l = 0

    def addFrame(self, frame):
        frame = self.frameToU16(frame)

        if (self.w is None) or (self.h is None):
            [self.h, self.w] = frame.shape
        else:
            assert(self.h == frame.shape[0])
            assert(self.w == frame.shape[1])

        frame.tofile(self.fp)
        self.l += 1
        
    def close(self):
        self.fp.close()

        self.w = int(self.w)
        self.h = int(self.h)
        
        inf_fp = open(self.root_name + ".inf", "w")
        inf_fp.write("binning = 1 x 1\n")
        inf_fp.write("data type = 16 bit integers (binary, little endian)\n")
        inf_fp.write("frame dimensions = " + str(self.w) + " x " + str(self.h) + "\n")
        inf_fp.write("number of frames = " + str(self.l) + "\n")
        inf_fp.write("Lock Target = 0.0\n")
        if True:
            inf_fp.write("x_start = 1\n")
            inf_fp.write("x_end = " + str(self.w) + "\n")
            inf_fp.write("y_start = 1\n")
            inf_fp.write("y_end = " + str(self.h) + "\n")
        inf_fp.close()

# split multi-channel images from DNA-FISH
def split_im_by_channels(
    im, 
    sel_channels, 
    all_channels, 
    single_im_size=default_im_size,
    num_buffer_frames=default_num_buffer_frames, 
    num_empty_frames=default_num_empty_frames, 
    skip_frame0=False,
    ):
    """Function to split a full image by channels"""
    _num_colors = len(all_channels)
    if isinstance(sel_channels, str) or isinstance(sel_channels, int):
        sel_channels = [sel_channels]
    _sel_channels = [str(_ch) for _ch in sel_channels]
    if isinstance(all_channels, str) or isinstance(all_channels, int):
        all_channels = [all_channels]
    _all_channels = [str(_ch) for _ch in all_channels]
    for _ch in _sel_channels:
        if _ch not in _all_channels:
            raise ValueError(f"Wrong input channel:{_ch}, should be within {_all_channels}")
    _ch_inds = [_all_channels.index(_ch) for _ch in _sel_channels]
    _ch_starts = [num_empty_frames + num_buffer_frames \
                    + (_i - num_empty_frames - num_buffer_frames) %_num_colors 
                    for _i in _ch_inds]
    #print('_ch_inds', _ch_inds)
    #print('_ch_starts', _ch_starts)
    if skip_frame0:
        for _i,_s in enumerate(_ch_starts):
            if _s == num_buffer_frames:
                _ch_starts[_i] += _num_colors

    _splitted_ims = [im[_s:_s+single_im_size[0]*_num_colors:_num_colors].copy() for _s in _ch_starts]

    return _splitted_ims