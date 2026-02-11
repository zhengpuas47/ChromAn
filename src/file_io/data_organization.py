import os, sys, re
import pandas as pd
import numpy as np
import warnings
from pathlib import Path


warnings.simplefilter(action='always', category=RuntimeWarning,)
warnings.simplefilter(action='once', category=UserWarning,)
color_usage_kwds = {
    'combo': 'c', 
    'decoded':'d',
    'unique': 'u', 
    'relabeled_combo':'l',
    'relabeled_unique':'v',
    'merfish': 'm', 
    'rna': 'r',
    'gene':'g',
    'protein':'p',
    }

_data_folder_reg = r'^H([0-9]+)[RQBUGCMP]([0-9]+)(.*)'
_data_fov_reg = r'(.+)_([0-9]+)\.(dax|tif|tiff)' # support dax, tif, tiff now

dax_regexp=r'H(?P<imagingRound>[0-9]+)[RMPCU]([0-9]+)/(?P<imageType>[a-zA-z_]+)_(?P<fov>[0-9]+).dax'
# Define standard regExp for confocal; 
confocal_regexp = r'([0-9_]+)/(?P<imageType>Conv|Confocal)_(Hyb|hyb)(?P<imagingRound>[0-9]+)_(?P<fov>[0-9]+)'
denoised_regexp = r'([0-9\+_BGM]+)/(?P<imageType>Batch Denoise_000_Conv|Batch Denoise_000_Confocal)_(Hyb|hyb)(?P<imagingRound>[0-9]+)_(?P<fov>[0-9]+)'
oligoIF_regexp = r'([0-9\+_BGM]+)/(?P<imageType>Conv|Batch Denoise_000_Confocal)_(Hyb|hyb)(?P<imagingRound>[0-9]+)_(?P<fov>[0-9]+)'

_default_DO_cols = ["channelName", "readoutName", "imageType", 
                    "imageRegExp", "bitNumber", "imagingRound", 
                    "color", "frame", "zPos", "fiducialImageType", 
                    "fiducialRegExp", "fiducialImagingRound", 
                    "fiducialFrame", "fiducialColor"]
_default_DO_fileRegExp = r'(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)'
_default_subfolder_fileRegExp = r'H(?P<imagingRound>[0-9]+)M([0-9]+)/(?P<imageType>[a-zA-z_]+)_(?P<fov>[0-9]+)'

def create_folder(_fd, verbose=True):
    """Formal create folder, resuable"""
    if os.path.exists(_fd) and not os.path.isdir(_fd):
        raise FileExistsError(f"target: {_fd} already exist and not a directory!")
    elif not os.path.exists(_fd):
        print(f"Creating folder: {_fd}")
        os.makedirs(_fd)
    else:
        print(f"Folder: {_fd} already exists")
    return

def generate_filemap(data_folder:Path, 
    regexp:str=confocal_regexp) -> pd.DataFrame:
    """Generate filemap dataframe for a given data folder, used for confocal generated nd2 data
    Args:
        data_folder (Path): path for this data; assuming the images are in subfolders of this path.
        regexp (str, optional): regular expression string. Defaults to confocal_regexp.

    Returns:
        fileMap (pandas.Dataframe): each row is one matched file, each column is each matched group.
    """
    # convert input:
    if not isinstance(data_folder, Path):
        data_folder = Path(data_folder)
        
    fileMap = {'imagePath':[]}

    for file_path in data_folder.rglob('*'):
        # re match
        relative_pathname = str(file_path).split(str(data_folder))[1].lstrip('/')
        match = re.match(regexp, relative_pathname)

        if file_path.is_file() and match:

            # aadd image Path:
            fileMap['imagePath'].append(relative_pathname)
            #print(relative_pathname)
            for _k, _v in match.groupdict().items():
                if _k not in fileMap:
                    fileMap[_k] = [_v]
                else:
                    fileMap[_k].append(_v)
    
    fileMap = pd.DataFrame(fileMap)
    # set data type
    if 'imagingRound' in fileMap.columns:
        fileMap = fileMap.astype({'imagingRound':int})
    if 'fov' in fileMap.columns:
        fileMap = fileMap.astype({'fov':int})
    
    return fileMap


def search_fovs_in_folders(
    master_folder, 
    folder_reg=_data_folder_reg,
    fov_reg=_data_fov_reg,
    verbose=True,
    ):
    """Function to get all subfolders"""
    # get sub_folders
    _folders = [os.path.join(master_folder, _f)
        for _f in os.listdir(master_folder) 
        if os.path.isdir(os.path.join(master_folder, _f)) and re.match(folder_reg, os.path.basename(_f))
    ]
    if len(_folders) == 0:
        raise ValueError(f"No valid subfolder detected, exit")
    # sort folders
    _folders = list(sorted(_folders, key=lambda _path:int(re.split(folder_reg, os.path.basename(_path) )[1]) ) ) 
    # scan for fovs
    _fovs = [_f for _f in os.listdir(_folders[0]) if re.match(fov_reg, _f)]
    # sort fovs
    _fovs = list(sorted(_fovs, key=lambda _path:int(re.split(fov_reg, os.path.basename(_path) )[2]) ) )     
    if verbose:
        print(f"- searching in folder: {master_folder}")
        print(f"-- {len(_folders)} folders, {len(_fovs)} fovs detected.")
    return _folders, _fovs


class Color_Usage(pd.DataFrame):
    
    def __init__(self, filename, verbose=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.verbose = verbose
        self.read_from_file()
        
    def read_from_file(self):
        """Load color_usage"""
        # read data from file and update the DataFrame
        if self.verbose:
            print(f"- load color_usage from file: {self.filename}")
        color_usage_df = pd.read_csv(self.filename, index_col=0)
        self.__dict__.update(color_usage_df.__dict__)

    def query(self, data_type, region_id):
        # define custom method here
        pass

    # query based on hyb_name
    def get_channel_info_for_round(self, hyb_name):
        # extract row
        _row = self.loc[hyb_name]
        # loop through to get rid of NaN
        _hyb_channels, _hyb_infos = [], []
        for _ch, _info in _row.items():
            #print(_info, type(_info))
            if isinstance(_info, str) or np.isfinite(_info):
                _hyb_channels.append(_ch)
                _hyb_infos.append(_info)
        return _hyb_channels, _hyb_infos
    # query based on data type and index
    def get_entry_location(
            self, 
            data_type, index, 
            dataType_kwds=color_usage_kwds,
            allow_duplicate=False,
        ):
        _key = f"{dataType_kwds[str(data_type).lower()]}{index}"
        _matches = []
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                if _info == _key:
                    _matches.append(
                        {'channel':_channel,'series':_hyb,}
                    )
        if len(_matches) == 0:
            warnings.warn("No matched entry detected", RuntimeWarning)
        elif len(_matches) > 1 and not allow_duplicate:
            warnings.warn("More than one match detected", RuntimeWarning)
        return _matches

    # a composite function to separate inds and channels for data-types
    def summarize_by_dataType(
            self, 
            dataType_kwds=color_usage_kwds,
            save_attrs=False,
            overwrite=False,
            allow_duplicate=False,
        ):
        # generate_regexp for given keywards
        if hasattr(self, '_dataType_2_ids') and \
        hasattr(self, '_dataType_2_channels') and \
        hasattr(self, '_dataType_2_hybs') and \
        not overwrite:
            return getattr(self, '_dataType_2_ids'), \
            getattr(self, '_dataType_2_channels'), \
            getattr(self, '_dataType_2_hybs')
        # summarize datatype
        _dataType_regexp = f"([{''.join(list(dataType_kwds.values()))}])([0-9]+)"
        # init
        _dataType_2_ids = {'others':[]}
        _dataType_2_channels = {'others':[]}
        _dataType_2_hybs = {'others':[]}
        # remaining names:
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                _match = re.match(str(_dataType_regexp), _info)
                if _match:
                    _dtype, _ind = _match.groups()
                    if _dtype not in _dataType_2_ids:
                        _dataType_2_ids[_dtype] = []
                        _dataType_2_channels[_dtype] = []
                        _dataType_2_hybs[_dtype] = []
                    # append
                    _dataType_2_ids[_dtype].append(int(_ind))
                    _dataType_2_channels[_dtype].append(_channel)
                    _dataType_2_hybs[_dtype].append(_hyb)
                elif _info != '' and _info != 'null' and _info != 'beads' and _info != 'empty': # ignore null, empty and beads
                    # only append new information, or allow_duplicate:
                    if allow_duplicate or _info not in _dataType_2_ids['others']:
                        # append
                        _dataType_2_ids['others'].append(_info)
                        _dataType_2_channels['others'].append(_channel)
                        _dataType_2_hybs['others'].append(_hyb)
                    
        # save
        if save_attrs:
            setattr(self, '_dataType_2_ids', _dataType_2_ids)
            setattr(self, '_dataType_2_channels', _dataType_2_channels)
            setattr(self, '_dataType_2_hybs', _dataType_2_hybs)

        return _dataType_2_ids, _dataType_2_channels, _dataType_2_hybs
    # for one folder, extract hybridization id
    def get_hyb_id(self, hyb, folder_regExp=_data_folder_reg):
        _match = re.match(folder_regExp, hyb)
        return int(_match.groups()[0])
        
    # method to extract info of special features:
    def get_info(self, info_query='PolyT'):
        _matches = []
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                if _info == info_query:
                    _matches.append(
                        {'channel':_channel,'series':_hyb,}
                    )
        if len(_matches) == 0:
            warnings.warn("No match detected", RuntimeWarning)
        return _matches
    
    # PolyT
    def get_polyt_info(self, polyt_query='PolyT'):
        return self.get_info(polyt_query)
    
    # DAPI
    def get_dapi_info(self, dapi_query='DAPI'):
        return self.get_info(dapi_query)
    
    # beads
    def get_fiducial_info(self, fiducial_query='beads'):
        return self.get_info(fiducial_query)

    @staticmethod
    def get_channels(color_usage_df):
        return list(color_usage_df.columns)
    @staticmethod
    def get_dapi_channel(color_usage_df, dapi_query='DAPI'):
        for _c in color_usage_df.columns:
            if dapi_query in color_usage_df[_c].fillna(-1).values:
                return _c
        return None
    @staticmethod
    def get_dapi_channel_index(color_usage_df):
        _dapi_ch = Color_Usage.get_dapi_channel(color_usage_df)
        if _dapi_ch is not None:
            return Color_Usage.get_channels(color_usage_df).index(_dapi_ch)
        else:
            return None
    @staticmethod
    def get_fiducial_channel(color_usage_df, fiducial_query='beads', use_dapi_if_not_found=True):
        for _c in color_usage_df.columns:
             if fiducial_query in color_usage_df[_c].fillna(-1).values:
                 return _c
        if use_dapi_if_not_found:
            return Color_Usage.get_dapi_channel(color_usage_df)
        return None
    @staticmethod
    def get_fiducial_channel_index(color_usage_df):
        _fiducial_ch = Color_Usage.get_fiducial_channel(color_usage_df)
        if _fiducial_ch is not None:
            return Color_Usage.get_channels(color_usage_df).index(_fiducial_ch)
        else:
            return None
    @staticmethod
    def get_imaged_channels(color_usage_df, hyb, ):
        image_infos = color_usage_df.iloc[color_usage_df.get_hyb_id(hyb)]
        imaged_channels = []
        for _channel, _info in image_infos.items():
            if isinstance(_info, float) and np.isnan(_info):
                continue
            elif isinstance(_info, str) and _info.lower() =='nan':
                continue
            else:
                imaged_channels.append(_channel)
        return imaged_channels
    @staticmethod
    def get_valid_channels(color_usage_df, hyb, dataType_kwds=color_usage_kwds,):
        image_infos = color_usage_df.iloc[color_usage_df.get_hyb_id(hyb)]
        valid_channels = []
        for _channel, _info in image_infos.items():
            if isinstance(_info, str) and _info[0] in list(dataType_kwds.values()):
                valid_channels.append(_channel)
        return valid_channels
    
## Create merlin version of data_organization
class Data_Organization(pd.DataFrame):
    """Class for Data_Organization in MERLin"""
    
    def __init__(self, 
        filename:str, 
        file_style:str='dax',
        verbose:bool=True, 
        *args, **kwargs):
        """
        Docstring for __init__
        
        :param self: Description
        :param filename: Description
        :type filename: str
        :param style: Description
        :type style: str
        :param verbose: Description
        :type verbose: bool
        :param args: Description
        :param kwargs: Description
        """

        # load from file if exist
        if os.path.isfile(filename):
            super().__init__(*args, **kwargs)
            self.filename = filename
            self.verbose = verbose
            self.read_from_file()            
        else:
            # create an empty data organization
            super().__init__(columns=_default_DO_cols, *args, **kwargs)
            self.filename = filename
            self.verbose = verbose
        
        if 'bitNumber' in self.columns:
            self['bitNumber'] = self['bitNumber'].astype(np.int32) # bitNumber should be int
            
        if file_style not in ['dax','nd2','tiff']:
            raise ValueError("style beyond dax, nd2 or tiff are not supported.")
        self.file_style = file_style
        
        return
    
    def read_from_file(self):
        """Load color_usage"""
        # read data from file and update the DataFrame
        if self.verbose:
            print(f"- load color_usage from file: {self.filename}")
        color_usage_df = pd.read_csv(self.filename, index_col=None)
        self.__dict__.update(color_usage_df.__dict__)
    
    def is_empty(self):
        return len(self) == 0

    def create_from_colorUsage(
        self, 
        color_usage_filename:str, 
        data_folder:str, 
        ref_Zstep:int, 
        readout_names:list=[],
        file_regExp:str=_default_DO_fileRegExp,
        dataType_kwds:dict=color_usage_kwds,
        zstep_size:float=0.5,        
        reorganized:bool=True, # whether subfolder is in format of H{hyb_id}R{round}
        verbose:bool=True,
    ):
        """Function to create a data_organization dataframe from colorUsage class
        Inputs:
        
        """
        if self.file_style == 'dax':
            from .dax_process import DaxProcesser
            # if file_regExp used default value and the data was not reorganized, change to subfolder_regExp
            if file_regExp == _default_DO_fileRegExp and not reorganized:
                file_regExp = _default_subfolder_fileRegExp
            
            _color_usage_df = Color_Usage(color_usage_filename)
            # search folders
            _, _fovs = search_fovs_in_folders(data_folder)
            # match with color_usage
            _dataType_2_ids, _dataType_2_channels, _dataType_2_hybs \
                = _color_usage_df.summarize_by_dataType(save_attrs=False)
            ## fill merfish related bits
            _merfish_feature = dataType_kwds['merfish']
            _ids, _channels, _hybs = _dataType_2_ids[_merfish_feature], \
                _dataType_2_channels[_merfish_feature], \
                _dataType_2_hybs[_merfish_feature]
            # check the datatype keywords:
            # loop through merfish rows
            for _ii, _i in enumerate(np.argsort(_ids)):
                
                _id, _channel, _hyb = _ids[_i], _channels[_i], _hybs[_i]
                print("MERFISHbit", _i, f'bit-{_id}', _channel, _hyb)
                # readouts
                if len(readout_names) == len(_ids):
                    _readout = readout_names[_ii] # always append readouts based on actual bit order
                else:
                    _readout = ''
                # for this hyb, try load the first fov as parameter reference:
                _test_filename = os.path.join(data_folder, _hyb, _fovs[0])
                if '.dax' in _test_filename:
                    _daxp = DaxProcesser(_test_filename, verbose=False)
                    _row = self._CreateRowSeries(
                        _id, _channel, _hyb, _readout, len(self)+1, 
                        _color_usage_df, _daxp, 
                        ref_Zstep, file_regExp, self.columns, channel_in_filename=reorganized)
                elif '.tif' in _test_filename or '.tiff' in _test_filename:
                    from tifffile import imread
                    _test_im = imread(_test_filename)
                    _hyb_channels, _hyb_infos = _color_usage_df.get_channel_info_for_round(_hyb)
                    _num_Zsteps = _test_im.shape[0] / len(_hyb_channels)
                    _row = self._CreateRowSeriesTiff(_id, _channel, _hyb, _readout, len(self)+1, 
                        _color_usage_df, 
                        _num_Zsteps, zstep_size, 
                        file_regExp, self.columns,
                        _filename_prefix='Conv_zscan')
                # append
                self.loc[len(self)] = _row
            if verbose:
                print(f"- {len(_ids)} MERFISH rows appended.")
            # everything except merfish: fill
            _other_infos, _other_channels, _other_hybs = \
                _dataType_2_ids['others'], \
                _dataType_2_channels['others'], \
                _dataType_2_hybs['others']
            for _info, _channel, _hyb in zip(_other_infos, _other_channels, _other_hybs):
                print("Other info:", _info, _channel, _hyb)
                _readout = '' # skip readouts
                # if dax file provided:
                _test_filename = os.path.join(data_folder, _hyb, _fovs[0])
                if '.dax' in _test_filename:
                    # for this hyb, try load the first fov as parameter reference:
                    _daxp = DaxProcesser(_test_filename, verbose=False)
                    # create
                    _row = self._CreateRowSeries(
                        _info, _channel, _hyb, _readout, len(self)+1, 
                        _color_usage_df, _daxp, 
                        ref_Zstep, file_regExp, self.columns, channel_in_filename=reorganized)
                elif '.tif' in _test_filename or '.tiff' in _test_filename:
                    from tifffile import imread
                    _test_im = imread(_test_filename)
                    _hyb_channels, _hyb_infos = _color_usage_df.get_channel_info_for_round(_hyb)
                    _num_Zsteps = _test_im.shape[0] / len(_hyb_channels)
                    _row = self._CreateRowSeriesTiff(_id, _channel, _hyb, _readout, len(self)+1, 
                        _color_usage_df, 
                        _num_Zsteps, zstep_size, 
                        file_regExp, self.columns,
                        _filename_prefix='Conv_zscan')
                # append
                self.loc[len(self)] = _row            
        
        elif self.file_style == 'nd2':
        # ND2 version:
            from .nd2_process import Nd2Processer
            # step1: create filemap:
            if file_regExp == _default_DO_fileRegExp:
                # reset this to be confocal:
                file_regExp = confocal_regexp
            # create filemap:
            filemap = generate_filemap(data_folder=data_folder,regexp=file_regExp)
            # load color_usage:
            _color_usage_df = Color_Usage(color_usage_filename)
            # summarize color_usage:
            _dataType_2_ids, _dataType_2_channels, _dataType_2_hybs \
                = _color_usage_df.summarize_by_dataType(save_attrs=False)
            # fill merfish related bits:
            _merfish_feature = dataType_kwds['merfish']
            _ids, _channels, _hybs = _dataType_2_ids[_merfish_feature], \
                _dataType_2_channels[_merfish_feature], \
                _dataType_2_hybs[_merfish_feature]
            # now based on images, 
            test_fov = filemap['fov'].unique()[0]
            fov_filemap = filemap.loc[filemap['fov']==test_fov]
            test_round = fov_filemap['imagingRound'].min()
            full_test_filename = os.path.join(str(data_folder), fov_filemap.loc[fov_filemap['imagingRound']==test_round,'imagePath'].values[0])
            # load an example ND2:
            _nd2_processer = Nd2Processer(full_test_filename)
            _nd2_processer._load_image()            
            # loop through merfish rows
            for _ii, _i in enumerate(np.argsort(_ids)):
                _id, _channel, _hyb = _ids[_i], _channels[_i], _hybs[_i]
                print("MERFISHbit", _i, f'bit-{_id}', _channel, _hyb)
                #print(fov_filemap.loc[_hyb in fov_filemap['imagePath'].values])
                
                _row = self._CreateRowSeriesND2(_id, _channel, _hyb, _ii+1, 
                                                readout_names[_ii], _color_usage_df, 
                                                fov_filemap, _nd2_processer=_nd2_processer)
                # append
                self.loc[len(self)] = _row
            # loop through other rows:
            # everything except merfish: fill
            _other_infos, _other_channels, _other_hybs = \
                _dataType_2_ids['others'], \
                _dataType_2_channels['others'], \
                _dataType_2_hybs['others']
            for _info, _channel, _hyb in zip(_other_infos, _other_channels, _other_hybs):
                print("Other info:", _info, _channel, _hyb)
                _readout = '' # skip readouts

                _row = self._CreateRowSeriesND2(_info, _channel, _hyb, len(self)+1, 
                                                _readout,_color_usage_df, 
                                                fov_filemap, _nd2_processer=_nd2_processer)
                # append
                self.loc[len(self)] = _row
        
        ## TODO: implement the tiff version;
        elif self.file_style == 'tiff':
            raise NotImplementedError("Tiff version was not implemented yet.")
        
        return
    
    # save
    def save_to_file(
        self,
        overwrite=False,
        verbose=True,
    ):
        """Function to save data_organization into CSV"""
        if not os.path.exists(self.filename) or overwrite:
            if verbose:
                print(f"Saving data_organization into file: {self.filename}.")
            self.to_csv(self.filename, index=None)
        else:
            if verbose:
                print(f"File: {self.filename} already exists, skip!")
    @staticmethod
    def _CreateRowSeries(_id, _channel, _hyb, 
                         _readout_name, _bit_num, 
                         _color_usage_df, _daxp, 
                         ref_Zstep, _file_regExp, _columns, channel_in_filename=True):
        """Frequently used function to convert info into pandas Series"""
        _zpos = _daxp._FindChannelZpositions(_daxp.xml_filename, verbose=False)[_channel]
        _frames = list(_daxp._FindChannelFrames(_daxp.filename, verbose=False)[_channel])
        _channels = _daxp._FindDaxChannels(_daxp.xml_filename, verbose=False)
        _fiducial_channel = _color_usage_df.get_fiducial_channel(_color_usage_df)
        
        if isinstance(_id, str):
            _bit_name = _id
        else:
            _bit_name = f'bit{_id}'
        # image_names:
        if channel_in_filename:
            _filename_prefix = '_'.join(_color_usage_df.get_channel_info_for_round(_hyb)[0]) + f"_s{_daxp.image_size[0]}"
        else:
            _filename_prefix = 'Conv_zscan'
        # ref-zstep
        if isinstance(ref_Zstep, int):
            _fiducial_frame_str = ref_Zstep*len(_channels) +_channels.index(_fiducial_channel)
        else:
            _fiducial_frame_str = '['+' '.join([str(_z) for _z in list(_daxp._FindChannelFrames(_daxp.filename, verbose=False)[_fiducial_channel])])+']'
        # prepare args
        _row = pd.Series([
            _bit_name, # bit name
            _readout_name, # readout name
            _filename_prefix,
            _file_regExp,
            _bit_num,
            _color_usage_df.get_hyb_id(_hyb),
            _channel,
            '['+' '.join([str(_z) for _z in _frames])+']',
            '['+' '.join([str(_z) for _z in _zpos])+']', 
            _filename_prefix,
            _file_regExp,
            _color_usage_df.get_hyb_id(_hyb),
            _fiducial_frame_str, #ref_Zstep*len(_channels) +_channels.index(_fiducial_channel),
            _fiducial_channel,  
        ], index=_columns)
        
        return _row
    @staticmethod
    def _CreateRowSeriesTiff(_id, _channel, _hyb, 
                             _readout_name, _bit_num, 
                             _color_usage_df, 
                             _num_Zsteps, _Zstep_size, 
                             _file_regExp, _columns,
                             _filename_prefix='Conv_zscan'):
        """Frequently used function to convert info into pandas Series"""
        #_zpos = _daxp._FindChannelZpositions(_daxp.xml_filename, verbose=False)[_channel]
        # assume centered at 0, generate zpos:
        _zpos = np.arange(_num_Zsteps)*_Zstep_size
        _zpos_str = '['+' '.join([str(_z) for _z in _zpos])+']'
        # get channels and infos from color_usage of this hyb:
        _channels, _infos = _color_usage_df.get_channel_info_for_round(_hyb)
        
        # get the index for this channel:
        if _channel not in _channels:
            raise ValueError(f"Channel: {_channel} not in the list of channels: {_channels}")
        _channel_index = _channels.index(_channel)
        # get frames
        _frames = np.array(np.arange(_num_Zsteps)*len(_channels) + _channel_index, dtype=np.int32)
        _frames_str = '['+' '.join([str(_z) for _z in _frames])+']'
        if isinstance(_id, str):
            _bit_name = _id
        else:
            _bit_name = f'bit{_id}'
        # fiducial:
        _fiducial_channel = _color_usage_df.get_fiducial_channel(_color_usage_df)
        _fiducial_channel_index = _color_usage_df.get_fiducial_channel_index(_color_usage_df) # get fiducial channels
        _fiducial_frames = np.array(np.arange(_num_Zsteps)*len(_channels) + _fiducial_channel_index, dtype=np.int32)
        _fiducial_frame_str = '['+' '.join([str(_z) for _z in _fiducial_frames])+']'
        #= '['+' '.join([str(_z) for _z in list(_daxp._FindChannelFrames(_daxp.filename, verbose=False)[_fiducial_channel])])+']'
        # prepare args
        _row = pd.Series([
            _bit_name, # bit name
            _readout_name, # readout name
            _filename_prefix,
            _file_regExp,
            _bit_num,
            _color_usage_df.get_hyb_id(_hyb),
            _channel,
            _frames_str,
            _zpos_str,
            _filename_prefix,
            _file_regExp,
            _color_usage_df.get_hyb_id(_hyb),
            _fiducial_frame_str, #ref_Zstep*len(_channels) +_channels.index(_fiducial_channel),
            _fiducial_channel,  
        ], index=_columns)
        
        return _row
    @staticmethod
    def _CreateRowSeriesND2(
        _id, _channel, _hyb, _bit_number,
        _readout_name, 
        _color_usage_df,
        _filemap,
        _nd2_processer,
        _file_regExp=confocal_regexp,
        _columns=_default_DO_cols,
    ):
        """Generate DataOrganization row series"""
        
        # channelName
        if isinstance(_id, str):
            _channelName = _id
        else:
            _channelName = f'bit{_id}' 
        _readoutName = str(_readout_name)
        _matched_row = _filemap.loc[[_hyb in _f for _f in _filemap['imagePath'].values]].iloc[0]
        _imageType = _matched_row['imageType']
        _imageRegExp = _file_regExp
        _bitNumber = int(_bit_number)
        _imagingRound = _matched_row['imagingRound']
        _color = str(_channel)
        _color_index = _nd2_processer.channel_indices[_nd2_processer.channels.index(_color)]
        _numZ = _nd2_processer._GetImageSize()['Z']
        _frame = np.arange(_numZ*_color_index, _numZ*(_color_index+1) )
        _frameStr = '['+' '.join([str(_f) for _f in _frame])+']'
        _zPos = np.round(_nd2_processer._FindChannelZpositions()[_color],2)
        _zPosStr = '['+' '.join([str(_f) for _f in _zPos])+']'
        _fiducialImageType = _imageType
        _fiducialRegExp = _imageRegExp
        _fiducialImageRound = _imagingRound
        # fiducial channel:
        _fiducialColor = str(_color_usage_df.get_fiducial_channel(_color_usage_df))
        _fiducial_color_index = _nd2_processer.channel_indices[_nd2_processer.channels.index(_fiducialColor)]
        _fiducialFrame = np.arange(_numZ*_fiducial_color_index, _numZ*(_fiducial_color_index+1) )
        _fiducialFrameStr = '['+' '.join([str(_f) for _f in _fiducialFrame])+']'
        # assemble:
        _row = pd.Series([
            _channelName, # bit name
            _readoutName, # readout name
            _imageType,
            _imageRegExp,
            _bitNumber,
            _imagingRound,
            _color,
            _frameStr,
            _zPosStr,
            _fiducialImageType,
            _fiducialRegExp,
            _fiducialImageRound,
            _fiducialFrameStr,
            _fiducialColor,  
        ], index=_columns)

        return _row
    
def find_zfill_number(data_folder, file_pattern="Conv_zscan_{fov}.dax"):
    """Function to find the number of digits in the file name"""
    for _z in range(10):
        _f = file_pattern.format(fov=str(0).zfill(_z))
        if os.path.isfile(os.path.join(data_folder, _f)):
            return _z
    
    raise ValueError(f"File: {file_pattern} not found in folder: {data_folder}")