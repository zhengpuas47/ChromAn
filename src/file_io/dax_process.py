import os, sys, re, time, h5py, pickle, sys
import numpy as np 
import xml.etree.ElementTree as ET
sys.path.append("..")
# required to load parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# default params
#from default_parameters import *
from default_parameters import default_num_buffer_frames,default_num_empty_frames,default_channels,default_ref_channel,default_im_size,default_dapi_channel
# usful functions
from correction_tools.load_corrections import load_correction_profile
from spot_tools.spot_class import Spots3D

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
        _xml_filename = filename.replace('.dax', '.xml') # xml file
        _channels = DaxProcesser._FindDaxChannels(_xml_filename, verbose=verbose)
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
                 RefCorrectionChannel=None,
                 SaveFilename=None,
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
        # verbose
        self.verbose = verbose
        # save filename
        if SaveFilename is None:
            self.save_filename = os.path.join(
                os.path.dirname(self.filename),
                os.path.basename(self.filename).split('.dax')[0] + '_processed.hdf5',
            )
        elif isinstance(SaveFilename, str):
            self.save_filename = SaveFilename
        else:
            raise TypeError("SaveFilename should be a string of file full path.")
        if os.path.isfile(self.save_filename):
            if self.verbose:
                print(f"- Existing save file: {self.save_filename}")
        else:
            if self.verbose:
                print(f"- New save file: {self.save_filename}")
        self.saving_log = {}
        # Correction folder
        self.correction_folder = CorrectionFolder

        # Channels
        if Channels is None:
            _loaded_channels = DaxProcesser._FindDaxChannels(self.xml_filename, verbose=self.verbose)
            if _loaded_channels is None:
                self.channels = default_channels
            else:
                self.channels = _loaded_channels
        elif isinstance(Channels, list) or isinstance(Channels, np.ndarray):
            self.channels = list(Channels)
        else:
            raise TypeError(f"Wrong input type for Channels")
        if FiducialChannel is not None and str(FiducialChannel) in self.channels:
            self.fiducial_channel = str(FiducialChannel)
        if DapiChannel is not None and str(DapiChannel) in self.channels:
            self.dapi_channel = str(DapiChannel)
        elif DapiChannel is None and default_dapi_channel in self.channels:
            self.dapi_channel = str(default_dapi_channel)
        if RefCorrectionChannel is not None and str(RefCorrectionChannel) in self.channels:
            self.ref_correction_channel = str(DapiChannel)
        elif RefCorrectionChannel is None and len(self.channels) > 1:
            self.ref_correction_channel = str(self.channels[1])
        # ImageSize
        try:
            self.image_size = DaxProcesser._FindImageSize(
                self.filename,
                channels=self.channels,
                verbose=False,
                )
        except:
            print("Not a typical image setting, image_size not determined.")
            raise Warning("Not a typical image setting, image_size not determined, auto-partition of channels are not possible. ")
        # Log for whether corrections has been done:
        self.correction_log = {_ch:{} for _ch in self.channels}
        self.correction_praram = {}

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
            self.loaded_channels = []
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
            self.num_buffer_frames = NbufferFrame
            self.num_empty_frames = NemptyFrame
            self.loaded_channels.extend(_loading_channels)
            # sort loaded
            self.loaded_channels = [_ch for _ch in sorted(self.loaded_channels, key=lambda v: self.channels.index(v))]
            return
        else:
            return _ims, _loading_channels
        
    def _calculate_drift(
        self, 
        RefImage, 
        FiducialChannel=None, 
        use_autocorr=True, 
        drift_kwargs={},
        save_attr=True, 
        save_ref_im=False,
        overwrite=False,
        ):
        """Calculate drift given reference image"""
        from correction_tools.alignment import align_image
        if hasattr(self, 'drift') and hasattr(self, 'drift_flag') and not overwrite:
            if self.verbose:
                print(f"- Drift already calculated, skip.")
            return self.drift, self.drift_flag
        # Load drift image
        if FiducialChannel is None and hasattr(self, 'fiducial_channel'):
            FiducialChannel = getattr(self, 'fiducial_channel')
        elif FiducialChannel is not None:
            FiducialChannel = str(FiducialChannel)
            if not hasattr(self, 'fiducial_channel'):
                self.fiducial_channel = FiducialChannel
        else:
            raise ValueError(f"Wrong input value for FiducialChannel: {FiducialChannel}")
        # Load drift image
        if self.correction_log[FiducialChannel].get('corr_drift', False):
            # if drift channel already warpped, stop
            if self.verbose:
                print(f"Fiducial image in channel {FiducialChannel} is broken, reload.")
            _DriftImage = self._load_image(sel_channels=[FiducialChannel], 
                                           ImSize=self.image_size,
                                           NbufferFrame=self.num_buffer_frames, NemptyFrame=self.num_empty_frames,
                                           save_attrs=False)[0][0]
        elif FiducialChannel in self.channels and hasattr(self, f"im_{FiducialChannel}"):
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

        # case1: RefImage is the same file:
        if isinstance(RefImage, str) and RefImage == self.filename:
            if RefImage == self.filename:
                if self.verbose:
                    print(f"-- processing ref_image itself, skip.")
                _drift = np.zeros(len(self.image_size))
                _drift_flag = -1
        else:
            # case2: RefImage is a different filename, load and get image:
            if isinstance(RefImage, str) and os.path.isfile(RefImage):
                # create class
                _dft_dax_cls = DaxProcesser(
                    RefImage, 
                    CorrectionFolder=self.correction_folder,
                    Channels=None, # only channels not specified
                    FiducialChannel=FiducialChannel,
                    DapiChannel=getattr(self, 'dapi_channel', None),
                    verbose=self.verbose
                    )
                # load image
                _dft_dax_cls._load_image(
                    sel_channels=[FiducialChannel], 
                    ImSize=self.image_size,
                    NbufferFrame=self.num_buffer_frames, 
                    NemptyFrame=self.num_empty_frames
                    )
                # get refimage
                RefImage = getattr(_dft_dax_cls, f"im_{FiducialChannel}")
            # case 3: a image is directly given, which matches size
            elif isinstance(RefImage, np.ndarray) and (np.array(RefImage.shape)==np.array(_DriftImage.shape)).all():
                pass # directly use
            else:
                raise ValueError(f"Wrong input of RefImage, should be either a matched sized image, or a filename")
            # save ref_im if specified
            if save_ref_im:
                self.ref_im = RefImage
            # align image
            _drift, _drift_flag = align_image(
                _DriftImage,
                RefImage, 
                use_autocorr=use_autocorr, 
                fiducial_channel=FiducialChannel,
                all_channels=self.channels,
                verbose=self.verbose, 
                **drift_kwargs,
            )
        # save attribute and return
        if save_attr:
            # drift channel
            self.fiducial_channel = FiducialChannel
            # drift results
            self.drift = _drift
            self.drift_flag = _drift_flag
        # return
        return _drift, _drift_flag

    # correct bleedthrough between channels
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
        from correction_tools.bleedthrough import bleedthrough_correction
        if correction_channels is None:
            correction_channels = self.loaded_channels
        # remove dapi channel:
        _correction_channels = [str(_ch) for _ch in correction_channels 
            if str(_ch) != getattr(self, 'dapi_channel', None)
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
            ref_channel=getattr(self, 'ref_correction_channel', None),
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
            self.correction_praram['corr_bleedthrough'] = {
                'rescale':rescale,
            }
            return 
        else:
            return _corrected_ims, _correction_channels
        
    # remove hot pixels
    def _corr_hotpixels(
        self, 
        correction_channels=None,
        correction_pf=None,
        correction_folder=None,
        hot_pixel_th:float=0.5, 
        hot_pixel_num_th:float=4,
        interpolation_style='nearest', 
        rescale=True, 
        save_attrs:bool=True,
        )->None:
        """Remove hot pixel by interpolation"""
        from correction_tools.filter import hot_pixel_correction
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
            if save_attrs:
                return 
            else:
                return [],[]
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_hot_pixel.")
        # process
        _corrected_ims = hot_pixel_correction(
            [getattr(self, f"im_{_ch}") for _ch in _correction_channels], 
            _correction_channels,
            correction_pf=correction_pf,
            correction_folder=correction_folder,
            hot_pixel_th=hot_pixel_th,
            hot_pixel_num_th=hot_pixel_num_th,
            interpolation_style=interpolation_style,
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
                self.correction_log[_ch]['corr_hot_pixel'] = True
            self.correction_praram['corr_hot_pixel'] = {
                'hot_pixel_th':hot_pixel_th, 
                'hot_pixel_num_th':hot_pixel_num_th,
                'interpolation_style':interpolation_style, 
                'rescale':rescale,
            }
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
        from correction_tools.illumination import illumination_correction
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
            if save_attrs:
                return
            else:
                return [],[]
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_illumination.")
        # load profile
        if correction_folder is None:
            correction_folder = self.correction_folder
        # process
        _corrected_ims = illumination_correction(
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
                self.correction_log[_ch]['corr_illumination'] = True
            self.correction_praram['corr_illumination'] = {
                'rescale':rescale,
            }
            return
        else:
            return _corrected_ims, _correction_channels

    # warp image
    def _corr_warpping_drift_chromatic(
        self,
        correction_channels=None,
        corr_drift=True,
        corr_chromatic=True, 
        correction_pf=None,
        correction_folder=None,
        ref_channel=None,
        warp_kwargs={'warp_order':1, 'border_mode':'grid-constant'},
        rescale=True, # not useful here
        save_attrs=True, 
        ):
        """Warp image in 3D to correct for translation and chromatic abbrevation
          this step require at least one of drift or chromatic profile.
          """
        from correction_tools.translate import warp_3D_images
        if not corr_chromatic and not corr_drift:
            raise ValueError("At least one of drift or chromatic should be specified")
        # get drift
        _drift = getattr(self, 'drift', np.zeros(len(self.image_size))) 
        # get channels
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        _chromatic_channels = [_ch for _ch in _correction_channels 
                            if _ch != getattr(self, 'fiducial_channel', None) and _ch != getattr(self, 'dapi_channel', None)]
        # check logs
        _drift_channels = [_ch for _ch in _correction_channels 
                           if _drift.any() # drift exist
                           #and _ch != getattr(self, 'fiducial_channel', None) # not fiducial channel
                           and not self.correction_log[_ch].get('corr_drift', False)] # have not corrected
        _chromatic_channels = [_ch for _ch in _correction_channels 
                               if _ch != getattr(self, 'fiducial_channel', None) # not fiducial channel
                               and not self.correction_log[_ch].get('corr_chromatic', False) # have not corrected
                               and corr_chromatic # need to specify
                               ] 
        # skip this if nothing specified
        if len(_drift_channels) == 0 and len(_chromatic_channels) == 0:
            if self.verbose:
                print(f"- Warp drift and chromatic already finished, skip. ")
            if save_attrs:
                return
            else:
                return [],[]
        # start warpping
        if self.verbose:
            print(f"- Start warpping images drift:{_drift_channels}, chromatic:{_chromatic_channels}")
        if correction_folder is None:
            correction_folder = self.correction_folder
        if ref_channel is None:
            ref_channel = getattr(self, 'ref_correction_channel', default_ref_channel)
        # process
        _corrected_ims = warp_3D_images(
            [getattr(self, f"im_{_ch}") for _ch in _correction_channels], 
            _correction_channels,
            corr_drift=(len(_drift_channels)>0),
            drift=_drift,
            drift_channels=_drift_channels,
            corr_chromatic=(len(_chromatic_channels)>0),
            correction_pf=correction_pf,
            chromatic_channels=_chromatic_channels,
            correction_folder=correction_folder,
            ref_channel=ref_channel,
            #rescale=rescale,
            verbose=self.verbose,
            **warp_kwargs,
        )
        # after finish, save attr
        if save_attrs:
            for _ch, _im in zip(_correction_channels, _corrected_ims):
                setattr(self, f"im_{_ch}", _im.copy())
            del(_corrected_ims)
            # update log
            for _ch in _correction_channels:
                if _ch in _drift_channels:
                    self.correction_log[_ch]['corr_drift'] = True
                if _ch in _chromatic_channels:
                    self.correction_log[_ch]['corr_chromatic'] = True
            self.correction_praram['warp'] = warp_kwargs
            self.correction_praram['warp']['rescale'] = rescale
            return
        else:
            return _corrected_ims, _correction_channels

    # generate chromatic functions
    def _corr_chromatic_functions(self, 
        correction_channels=None,
        correction_pf=None, 
        correction_folder=None,
        ref_channel=None,
        save_attrs=True,
        overwrite=False,
        ):
        """Generate chromatic_abbrevation functions for each channel"""
        from correction_tools.chromatic import generate_chromatic_function
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
        # ref channel
        if ref_channel is None:
            ref_channel = getattr(self, 'ref_correction_channel', default_ref_channel)
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
        
    # Gaussian highpass for high-background images
    def _corr_gaussian_highpass(
        self,                            
        correction_channels=None,
        correction_pf=None,
        correction_folder=None,
        sigma=3,
        truncate=2,
        rescale=True, 
        save_attrs=True,
        ):
        """Function to apply gaussian highpass for selected channels"""
        from correction_tools.filter import gaussian_highpass_correction
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
            if save_attrs:
                return 
            else:
                return [],[]
        # update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for gaussian_highpass.")
        # loop through channels
        # process
        _corrected_ims = gaussian_highpass_correction(
            [getattr(self, f"im_{_ch}") for _ch in _correction_channels], 
            _correction_channels,
            correction_pf=correction_pf,
            correction_folder=correction_folder,
            sigma=sigma,
            truncate=truncate,
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
                self.correction_log[_ch]['corr_gaussian_highpass'] = True
            self.correction_praram['corr_gaussian_highpass'] = {
                'sigma':sigma, 
                'truncate':truncate,
                'rescale':rescale, 
            }
            return 
        else:
            return _corrected_ims, _correction_channels
        
    # Spot_fitting:
    def _fit_3D_spots(
        self,
        fit_channels:list=None,
        channel_2_seeding_kwargs:dict=dict(),
        #seeding_kwargs:dict=dict(),
        fitting_mode:str='cpu',
        channel_2_seeds:dict=dict(),
        channel_2_fitting_kwargs:dict=dict(),
        normalization:str=None, 
        normalization_kwargs:dict=dict(),
        overwrite:bool=False,
        detailed_verbose:bool=False,        
    ):
        """Function to call spot_fitting module to perform 3D spot fitting.
        Inputs:
            fit_channels: list of channels to be fit, list or np.ndarray,
            seeding_kwargs: key arguments for seeding, dict,
            fitting_mode: type of fitting to be performed, str (default:'cpu'),
            normalization: type of normalization to fitted spot intensities, str (default:None),
            normalization_kwargs: key arguments for normalization, dict,
            overwrite: whether overwrite existing fitted spots, bool (default:False),
            detailed_verbose: print more details, bool (default:False),
        Outputs:
            fitted_spots: list of Spots3D for fitted channels,
            fitted_channels: list of channels that actually performed fitting.
        """
        # determine channels to fit
        from spot_tools.spot_fitting import SpotFitter
        if fit_channels is None:
            _fit_channels = self.loaded_channels
        elif isinstance(fit_channels, list) or isinstance(fit_channels, np.ndarray):
            _fit_channels = [str(_ch) for _ch in fit_channels]
        else:
            raise TypeError(f"Wrong input type ({type(fit_channels)}) for fit_channels.")
        # check seeding_kwargs_lsit and fitting_kwargs_list to match the length 
        
        
        _fit_channels = [str(_ch) for _ch in _fit_channels 
            if str(_ch) != getattr(self, 'fiducial_channel', None) \
                and str(_ch) != getattr(self, 'dapi_channel', None)
            ] # not include fiducial and dapi channels
        ## all finished, directly return:
        _require_fitting_flags = [
            (not hasattr(self, f"spots_{_ch}") or overwrite) \
            and hasattr(self, f"im_{_ch}") 
            for _ch in _fit_channels
        ]
        
        # modify fitted channels
        _fit_channels = [_ch 
                         for _ch, _flag in zip(_fit_channels, _require_fitting_flags)
                         if _flag]
        _fit_spots = []
        if not hasattr(self, 'fitting_log'):
            self.fitting_log = {_ch:{} for _ch in self.channels}        
        # start
        for _ch in _fit_channels:
            # get channel-specific kwargs:
            _seeding_kwargs = channel_2_seeding_kwargs.get(_ch, dict())
            _fitting_kwargs = channel_2_fitting_kwargs.get(_ch, dict())
            _pre_seeds = channel_2_seeds.get(_ch, None) # pre-selected seeds
            
            if self.verbose and not detailed_verbose:
                print(f"-- fit spots in channel: {_ch}", end=', ')
                _fit_start = time.time()
            _fitter = SpotFitter(
                getattr(self, f"im_{_ch}"),
                _seeding_kwargs,
                _fitting_kwargs,
                detailed_verbose,
            )
            # seeding 
            if not isinstance(_pre_seeds, np.ndarray) or len(_pre_seeds.shape) != 2: # pre-seed has to be 2d array
                _fitter.seeding()
            else:
                _fitter.seeds = _pre_seeds
            # fitting
            if fitting_mode == 'cpu':
                _fitter.CPU_fitting(
                    remove_boundary_points=False,
                    normalization=normalization,
                    normalization_kwargs=normalization_kwargs,
                )
            # retrive spots
            _spots = _fitter.spots.copy()
            # add to attribute
            setattr(self, f"spots_{_ch}", _spots)
            _fit_spots.append(_spots)
            # save log
            self.fitting_log[_ch].update(
                {
                    'seeding': _fitter.seeding_parameters,
                    'fitting': _fitter.fitting_parameters,
                    'remove_boundary_points': False,
                    'normalization': normalization,
                    'normalization_kwargs': normalization_kwargs, 
                }
            )
            if self.verbose and not detailed_verbose:
                print(f"{len(_spots)} fitted in {time.time()-_fit_start:.3f}s.")
        # return
        return _fit_spots, _fit_channels

    # Saving:
    def _save_param_to_hdf5(
      self,
      save_type:str,
      hdf5_filename:str=None, 
      key:str=None,
      overwrite:bool=False,
    ):
        """
        Function to save correction or fitting parameters to hdf5.
        Inputs:
            save_type: correction or fitting, str;
            hdf5_filename: full filename of hdf5 target save file, str;
            key: save key within this hdf5 file, str;
            overwrite: whether overwrite existing datasets in hdf5,
        """
        # check save_type
        if save_type in ['correction', 'fitting']:
            pass
        elif save_type in ['spots', 'im']:
            raise ValueError(f"For data, use _save_data_to_hdf5. ")
        elif save_type in ['base']:
            raise ValueError(f"For data, use _save_base_to_hdf5. ")
        else:
            raise ValueError("Invalid save_type.")
        # get default save information:
        if hdf5_filename is None:
            if self.verbose:
                print("- use default save filename.")
            hdf5_filename = self.save_filename
        if key is None:
            if self.verbose:
                print("- use default save key.")
            key = save_type
        # retrieve information:
        _param_dict = getattr(self, f"{save_type}_log")
        with h5py.File(hdf5_filename, 'a') as _f:
            for _ch, _info in _param_dict.items():
                _ch_key = key + '/' + _ch
                _channel_updated_attrs = []
                if _ch_key not in _f or overwrite:
                    _group = _f.require_group(_ch_key)
                else:
                    _group = _f[_ch_key]
                # save attrs
                for _k,_v in _info.items():
                    if _k not in _group.attrs or overwrite:
                        _group.attrs[_k] = _v
                        _channel_updated_attrs.append(_k)
                if self.verbose:
                    print(f"-- saved {_ch_key} with {_channel_updated_attrs} attributes.")

        return
    def _save_base_to_hdf5(
        self,
        hdf5_filename:str=None, # full filename of the hdf5 
        key:str='.', # location to save attrs, by default at the root of the file.
        overwrite:bool=False, # whether overwrite attrs in the target file
        ):
        """
        Function to save basic information into hdf5.
        Inputs:
            save_type: correction or fitting, str;
            hdf5_filename: full filename of hdf5 target save file, str;
            overwrite: whether overwrite existing datasets in hdf5,
        """
        # collect all arguments to be saved
        _sel_attrs = [
            'filename', 'inf_filename', 'off_filename', 'power_filename', 'xml_filename', # filenames
            'save_filename', # save information
            'correction_folder', # correction basic info
            'channels', 'fiducial_channel', 'dapi_channel', 'ref_correction_channel', # channels
        ]
        # get default save information:
        if hdf5_filename is None:
            if self.verbose:
                print("- use default save filename.")
            hdf5_filename = self.save_filename
        # get default save information:
        if key is None:
            if self.verbose:
                print("- use default save key.")
            key = "." # root
        # get save target file:
        if not os.path.exists(os.path.dirname(hdf5_filename)):
            if self.verbose:
                print(f"Creating folder: {os.path.dirname(hdf5_filename)}")
            os.makedirs(os.path.dirname(hdf5_filename))
        # open this file:
        if self.verbose:
            if os.path.exists(hdf5_filename):
                print(f"- saving to existing file: {hdf5_filename}")
            else:
                print(f"- saving to new file: {hdf5_filename}")
        # start saving:
        _saved_attrs = []
        with h5py.File(hdf5_filename, 'a') as _f:
            # create this group:
            _g = _f.require_group(key)
            # loop through attributes
            for _attr in _sel_attrs:
                if hasattr(self, _attr) and getattr(self, _attr) is not None:
                    if _attr not in _g.attrs or _g.attrs[_attr] is None or overwrite:
                        print(_attr)
                        _g.attrs[_attr] = getattr(self, _attr)
                        _saved_attrs.append(_attr)
        if self.verbose:
            if len(_saved_attrs) > 0:
                print(f"-- updated the following basic information: {','.join(_saved_attrs)}")
            else: 
                print(f"-- all attributes exist, skip.")

    def _save_data_to_hdf5(
        self,
        channel:str, # one channel
        save_type:str, # 'spots'|'im'
        hdf5_filename:str=None, 
        key=None,
        index=None,
        compression='gzip',
        overwrite=False,
        ):
        """Function to save selected information to given hdf5 file and key
        Inputs:
            channel: color channel to save, str;
            save_type: datatype to be saved, str ([spots, im]);
            hdf5_filename: full filename of hdf5 target save file, str;
            key: save key within this hdf5 file, str;
            index: index within this hdf5_filename/key dataset, int or slice;
            compression: type of compression applied to hdf5, str or None;
            overwrite: whether overwrite existing datasets in hdf5,
        NOTICE: this method doesn't allow change of dataset shape in hdf5 file.
        """
        # check channel
        if channel not in self.channels:
            raise ValueError(f"Wrong channel:{channel}")
        # check if info exists:
        if save_type in ['spots', 'im']:
            _attr_name = f"{save_type}_{channel}"
            if not hasattr(self, _attr_name):
                raise AttributeError(f"Target data: {_attr_name} doesn't exist.")
        elif save_type in ['base', 'correction', 'fitting']:
            raise ValueError(f"For parameters, use save_param_to_hdf5. ")
        else:
            raise ValueError("Invalid save_type.")
        if save_type not in self.saving_log:
            self.saving_log[save_type] = []
        # get default save information:
        if hdf5_filename is None:
            if self.verbose:
                print("- use default save filename.")
            hdf5_filename = self.save_filename
        if key is None:
            if self.verbose:
                print("- use default save key.")
            key = '/'.join([channel, save_type])
        # get save target file:
        if not os.path.exists(os.path.dirname(hdf5_filename)):
            if self.verbose:
                print(f"Creating folder: {os.path.dirname(hdf5_filename)}")
            os.makedirs(os.path.dirname(hdf5_filename))
        # open this file:
        if self.verbose:
            if os.path.exists(hdf5_filename):
                print(f"- saving to existing file: {hdf5_filename}")
            else:
                print(f"- saving to new file: {hdf5_filename}")
        # open and search key
        _data = getattr(self, _attr_name)
        with h5py.File(hdf5_filename, 'a') as _f:
            if key not in _f or overwrite:
                if index is None:
                    _dataset = _f.require_dataset(key, 
                                                  shape=_data.shape, 
                                                  dtype=_data.dtype,
                                                  chunks=_data.shape,
                                                  compression=compression,
                                                  )
                    _dataset[:] = _data
                else:
                    _dataset = _f.require_dataset(key, 
                                                  shape=tuple([index]+list(_data.shape)), 
                                                  dtype=_data.dtype,
                                                  chunks=[1]+list(_data.shape),
                                                  compression=compression,
                                                  )
                    _dataset[index] = _data
                if self.verbose:
                    print(f"-- saving {key}, shape={_data.shape}")
                # update log
                self.saving_log[save_type].append(
                    [save_type, channel, hdf5_filename, key, index, True]
                )
            else:
                if self.verbose:
                    print(f"-- skip saving {key}, already exists")
                # write to log
                self.saving_log[save_type].append(
                    [save_type, channel, hdf5_filename, key, index, False]
                )
        return

    def _save_to_npy(self, save_channels, save_folder=None, save_basename=None, overwrite=False):
        
        if save_folder is None:
            save_folder = os.dirname(self.save_filename)
        if not os.path.exists(save_folder):
            if self.verbose:
                print("Create folder: ", save_folder)
            os.makedirs(save_folder)
        # basename
        if save_basename is None:
            save_basename = os.path.basename(self.filename).split(os.extsep)[0]
        for _ch in save_channels:
            _im = getattr(self, f"im_{_ch}")
            _channel_save_filename = os.path.join(save_folder, f"{save_basename}_{_ch}.npy")
            if overwrite or not os.path.exists(_channel_save_filename):
                if self.verbose:
                    print(f"-- save channel {_ch} to {_channel_save_filename}")
                np.save(_channel_save_filename, _im)                   

    # Loading:
    def _load_from_hdf5(self, channel, type, hdf5_filename, key):
        # TODO: write proper load_from_hdf5
            pass
    
    @staticmethod
    def _FindShutterStr(
        xml_filename,
        ):
        """Find shutter filename"""
        _xml_filename = xml_filename
        try:
            _hal_info = ET.parse(_xml_filename).getroot()
            _shutter_filename = _hal_info.findall('illumination/shutters')[0].text
            return _shutter_filename        
        except:
            return None
    @staticmethod
    def _FindDaxChannels(xml_filename,
                         verbose=True,
                         ):
        """Find channels"""
        _xml_filename = xml_filename
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
    def _FindTotalNumFrame(inf_filename):
        return int(DaxProcesser._LoadInfFile(inf_filename)['number of frames'])
    @staticmethod
    def _FindImageSize(dax_filename, 
                       channels=None,
                       NbufferFrame=default_num_buffer_frames,
                       verbose=True,
                       ):
        _inf_filename = dax_filename.replace('.dax', '.inf') # info file
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        if channels is None:
            channels = DaxProcesser._FindDaxChannels(_xml_filename)                
        # get frame information from .inf file
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
    @staticmethod
    def _FindChannelZpositions(
        xml_filename,
        verbose=True,
        ):
        import xml.etree.ElementTree as ET
        """Find Z positions from xml file"""
        _xml_filename = xml_filename # xml file
        _inf_filename = xml_filename.replace('.xml', '.inf')
        #try:
        _hal_info = ET.parse(_xml_filename).getroot()
        _zpos_string = _hal_info.findall('focuslock/hardware_z_scan/z_offsets')[0].text
        _zpos = np.array(_zpos_string.split(','), dtype=np.float32)
        # get channels
        _channels = DaxProcesser._FindDaxChannels(_xml_filename, verbose=False)
        if len(_zpos) != int(DaxProcesser._LoadInfFile(_inf_filename)['number of frames']):
            raise ValueError("Z position number doesn't match total image length.")
        if int(len(_zpos) / len(_channels)) != len(_zpos) / len(_channels):
            raise ValueError("Z position number doesn't match channels.")
        # otherwise, proceed to parse:
        _ch_2_zpos = {_ch:[] for _ch in _channels}
        for _i, _z in enumerate(_zpos):
            _ch_2_zpos[_channels[_i%len(_channels)]].append(_z)
        _ch_2_zpos = {_ch:np.array(_zs,) 
                      for _ch, _zs in _ch_2_zpos.items()}
        if verbose:
            print(f"-- z positions for channel: {list(_ch_2_zpos.keys())} found")
        return _ch_2_zpos
    @staticmethod
    def _FindChannelFrames(
        dax_filename,
        verbose=True,
    ):
        """Find frame number for each channel"""
        _inf_filename = dax_filename.replace('.dax', '.inf') # info file
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        # get total number of frames
        _total_frame_num = DaxProcesser._FindTotalNumFrame(_inf_filename)
        # get channels
        _channels = DaxProcesser._FindDaxChannels(_xml_filename, verbose=False)
        # check inputs:
        if int(_total_frame_num / len(_channels)) != _total_frame_num / len(_channels):
            raise ValueError("Total number of frames doesn't match channels.")
        # process
        _ch_2_frames = {_ch:[] for _ch in _channels}
        for _i in range(_total_frame_num):
            _ch_2_frames[_channels[_i%len(_channels)]].append(_i)
        _ch_2_frames = {_ch:np.array(_zs, dtype=np.int32) 
                        for _ch, _zs in _ch_2_frames.items()}
        if verbose:
            print(f"-- Frame inds for channel: {list(_ch_2_frames.keys())} found")
        return _ch_2_frames
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
    @staticmethod
    def _RunDapiSegmentation3D(dapi_image:np.ndarray,
                               fov_id=None,
                               model_type='nuclei',
                               use_gpu=True,
                               segmentation_kwargs:dict=dict(), 
                               subsampling=False, subsampling_ratio=0.5,
                               save=False, segmentation_filename='dapi_3d_labels.npy',
                               verbose=True,
                               ):
        """Function to run Cellpose segmentation segmentation for DAPI image"""
        from cellpose import models
        from torch.cuda import empty_cache
        _default_segmentation_kwargs = {
            'diameter':100,
            'anisotropy': 4.673, #500nm/107nm, ratio of z-pixel vs x-y pixels
            'channels':[0,0],
            'min_size':2000,
            'do_3D': True,
            'cellprob_threshold': 0,
        }
        _default_segmentation_kwargs.update(segmentation_kwargs)
        if subsampling:
            import cv2
            # subsampled size:
            _subsampled_sizes = list(dapi_image.shape) 
            _subsampled_sizes[-2] = int(_subsampled_sizes[-2] * subsampling_ratio)
            _subsampled_sizes[-1] = int(_subsampled_sizes[-1] * subsampling_ratio)
            _dapi_image = np.array([cv2.resize(_ly, (_subsampled_sizes[-2], _subsampled_sizes[-1])) 
                         for _ly in dapi_image])
            _default_segmentation_kwargs['diameter'] = _default_segmentation_kwargs['diameter'] * subsampling_ratio
            _default_segmentation_kwargs['anisotropy'] = _default_segmentation_kwargs['anisotropy'] * subsampling_ratio
        else:
            _dapi_image = dapi_image
        # Create cellpose model
        if verbose:
            print(f"- run Cellpose segmentation", end=' ')
            _cellpose_start = time.time()
        #empty_cache() # empty cache to create new model
        seg_model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        
        # Run cellpose prediction
        _seg_label, _, _ = seg_model.eval(np.stack([_dapi_image,_dapi_image], axis=3), 
                                        *_default_segmentation_kwargs,
                                        )
        if subsampling:
            _seg_label = np.array([cv2.resize(_ly, dapi_image.shape[-2:], 
                                interpolation=cv2.INTER_NEAREST_EXACT) 
                                for _ly in _seg_label], dtype=np.int32)
        if verbose:
            print(f"in {time.time()-_cellpose_start:.3f}s.")
        # save
        if save:
            if verbose:
                print(f"-- saving segmentation labels into file: {segmentation_filename}")
            # save
        if segmentation_filename.split(os.extsep)[-1] == 'npy':
            np.save(segmentation_filename, _seg_label)
        elif segmentation_filename.split(os.extsep)[-1] == 'pkl':
            pickle.dump(_seg_label, open(segmentation_filename, 'wb'))
        elif segmentation_filename.split(os.extsep)[-1] == 'hdf5' or segmentation_filename.split(os.extsep)[-1] == 'h5':
            with h5py.File(segmentation_filename, 'a') as _f:
                if len(_f.keys()) == 0:
                    _target = _f
                else:
                    if fov_id is None:
                        fov_id = list(_f.keys())[0]
                    _target = _f[str(fov_id)]
                # create
                _target.create_dataset('dna_mask', data=_seg_label)
        # return
        return _seg_label
    # Composite functions to run correction:
    def RunCorrection(self, 
                      correction_folder=None,
                      correction_channels=None,
                      correction_pf_dict=dict(),
                      corr_hotpixel=False,
                      corr_hotpixel_params=dict(),
                      corr_bleed=True,
                      corr_bleed_params=dict(),
                      corr_illumination=True,
                      corr_illumination_params=dict(),
                      corr_chromatic=True,
                      corr_chromatic_params=dict(),
                      corr_drift=True,
                      ref_fiducial_image=None,
                      drift_params=dict(),
                      warp_image=False,
                      warp_params=dict(),
                      ):
        """Function to systematically run correction"""
        # check correction_folder
        if correction_folder is None:
            if self.correction_folder is None:
                raise ValueError("No correction_folder specified.")
            else:
                correction_folder = self.correction_folder
        # check correction_channels
        if correction_channels is None:
            if self.channels is None:
                raise ValueError("No correction_channels specified.")
            else: 
                correction_channels = [_ch for _ch in self.channels] 
                if hasattr(self, 'dapi_channel'):
                    correction_channels = [_ch for _ch in self.channels if _ch != self.dapi_channel] # exclude DAPI      
        # perform corrections:
        # 0. correct hotpixels
        if corr_hotpixel:
            if self.verbose:
                print("- run hotpixel correction")
            self._corr_hotpixels(correction_channels=correction_channels,
                                  correction_folder=correction_folder,
                                  correction_pf=correction_pf_dict.get('hot_pixel',None),
                                  **corr_hotpixel_params,
                                  )
        # 1. bleed correction
        if corr_bleed:
            if self.verbose:
                print("- run bleed correction")
            self._corr_bleedthrough(correction_channels=[_ch for _ch in self.channels 
                                                         if hasattr(self, 'dapi_channel') and str(_ch) != self.dapi_channel],
                                    correction_folder=correction_folder,
                                    correction_pf=correction_pf_dict.get('bleedthrough',None),
                                    **corr_bleed_params,
                                    )
        # 2. illumination correction
        if corr_illumination:
            if self.verbose:
                print("- run illumination correction")
            self._corr_illumination(correction_channels=correction_channels,
                                    correction_folder=correction_folder,
                                    correction_pf=correction_pf_dict.get('illumination',None),
                                    **corr_illumination_params,
                                    )
        # 3 calculate drift
        if corr_drift:
            if self.verbose:
                print("- run drift correction")
            self._calculate_drift(RefImage=ref_fiducial_image,
                                  **drift_params,
                                  )
        # 4. chromatic correction
        # if warp:
        if warp_image:
            if self.verbose:
                print("- run warp correction")
            self._corr_warpping_drift_chromatic(
                correction_channels=correction_channels,
                corr_drift=corr_drift,
                corr_chromatic=corr_chromatic,
                correction_folder=correction_folder,
                correction_pf=correction_pf_dict.get('chromatic',None),
                **warp_params)
        else:
            if corr_chromatic:
                if self.verbose:
                    print("- run chromatic correction")
                self._corr_chromatic_functions(correction_channels=correction_channels,
                                               correction_folder=correction_folder,
                                               correction_pf=correction_pf_dict.get('chromatic_function',None),
                                               **corr_chromatic_params,
                                               )
        # return
        return
    
    
    
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

# slurm in 