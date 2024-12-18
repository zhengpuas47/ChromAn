import os, sys, re
import pandas as pd
import numpy as np
import warnings
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
    
    ### TODO: add query functions for color_usage
    def summarize(
        self,
        overwrite:bool=True,
        ):
        """get a summary of the full color usage, sort by data_type:"""
        if hasattr(self, 'dataTypeDict') and not overwrite:
            if self.verbose:
                print(f"summary already exists, skip.")
            return
        dataType_2_infoDict = {}
        for _data_type, _data_key in color_usage_kwds.items():
            for _i, _row in self.iterrows():
                pass

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
                        {'channel':_channel,'hyb':_hyb,}
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
        _dataType_2_ids = {}
        _dataType_2_channels = {}
        _dataType_2_hybs = {}
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
                    if 'others' not in _dataType_2_ids:
                        _dataType_2_ids['others'] = []
                        _dataType_2_channels['others'] = []
                        _dataType_2_hybs['others'] = []
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
    # PolyT
    def get_polyt_info(self, polyt_query='PolyT'):
        _polyt_matches = []
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                if _info == polyt_query:
                    _polyt_matches.append(
                        {'channel':_channel,'hyb':_hyb,}
                    )
        if len(_polyt_matches) == 0:
            warnings.warn("No polyT match detected", RuntimeWarning)
        return _polyt_matches
    # DAPI
    def get_dapi_info(self, dapi_query='DAPI'):
        _dapi_matches = []
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                if _info == dapi_query:
                    _dapi_matches.append(
                        {'hyb':_hyb, 'channel':_channel,}
                    )
        if len(_dapi_matches) == 0:
            warnings.warn("No DAPI match detected", RuntimeWarning)
        return _dapi_matches
    # beads
    def get_fiducial_info(self, fiducial_query='beads'):
        _fiducial_matches = []
        for _hyb in self.index:
            _channels, _infos = self.get_channel_info_for_round(_hyb)
            # loop through channels
            for _channel, _info in zip(_channels, _infos):
                if _info == fiducial_query:
                    _fiducial_matches.append(
                        {'hyb':_hyb, 'channel':_channel,}
                    )
        if len(_fiducial_matches) == 0:
            warnings.warn("No fiducial match detected", RuntimeWarning)
        return _fiducial_matches

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
    def get_fiducial_channel(color_usage_df, fiducial_query='beads'):
        for _c in color_usage_df.columns:
             if fiducial_query in color_usage_df[_c].fillna(-1).values:
                 return _c
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
                 verbose:bool=True, 
                 *args, **kwargs):
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
            # create

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