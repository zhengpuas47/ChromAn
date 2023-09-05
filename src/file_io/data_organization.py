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
_data_fov_reg = r'(.+)_([0-9]+)\.dax'

_default_DO_cols = ["channelName", "readoutName", "imageType", 
                    "imageRegExp", "bitNumber", "imagingRound", 
                    "color", "frame", "zPos", "fiducialImageType", 
                    "fiducialRegExp", "fiducialImagingRound", 
                    "fiducialFrame", "fiducialColor"]
_default_DO_fileRegExp = '(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)'

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
        sel_feature_ind:int=0,
        verbose:bool=True,
    ):
        """Function to create a data_organization dataframe from colorUsage class
        Inputs:
        
        """
        from .dax_process import DaxProcesser
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
        print(_ids)
        # loop through merfish rows
        for _i in np.argsort(_ids):
            _id, _channel, _hyb = _ids[_i], _channels[_i], _hybs[_i]
            # readouts
            if len(readout_names) == len(_ids):
                _readout = readout_names[_i]
            else:
                _readout = ''
            # for this hyb, try load the first fov as parameter reference:
            _daxp = DaxProcesser(os.path.join(data_folder, _hyb, _fovs[0]), verbose=False)
            # create
            _row = self._CreateRowSeries(
                _id, _channel, _hyb, _readout, len(self)+1, 
                _color_usage_df, _daxp, 
                ref_Zstep, file_regExp, self.columns)
            # append
            self.loc[len(self)] = _row
        if verbose:
            print(f"- {len(_ids)} MERFISH rows appended.")
        ## fill polyT and DAPI
        # polyt:
        _polyt_info = _color_usage_df.get_polyt_info()
        if len(_polyt_info) > 0:
            _polyt_info = _polyt_info[sel_feature_ind]
            _id, _channel, _hyb = 'PolyT', _polyt_info['channel'], _polyt_info['hyb']
            # for this hyb, try load the first fov as parameter reference:
            _daxp = DaxProcesser(os.path.join(data_folder, _hyb, _fovs[0]), verbose=False)
            # create
            _row = self._CreateRowSeries(
                _id, _channel, _hyb, 'polyt', len(self)+1, 
                _color_usage_df, _daxp, 
                ref_Zstep, file_regExp, self.columns)
            # append
            self.loc[len(self)] = _row
            if verbose:
                print(f"- PolyT row appended.")
        # dapi
        _dapi_info = _color_usage_df.get_dapi_info()
        if len(_dapi_info) > 0:
            _dapi_info = _dapi_info[sel_feature_ind]
            _id, _channel, _hyb = 'DAPI', _dapi_info['channel'], _dapi_info['hyb']
            # for this hyb, try load the first fov as parameter reference:
            _daxp = DaxProcesser(os.path.join(data_folder, _hyb, _fovs[0]), verbose=False)
            # create
            _row = self._CreateRowSeries(
                _id, _channel, _hyb, 'dapi', len(self)+1, 
                _color_usage_df, _daxp, 
                ref_Zstep, file_regExp, self.columns)
            # append
            self.loc[len(self)] = _row
            if verbose:
                print(f"- DAPI row appended.")            
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
                         ref_Zstep, _file_regExp, _columns):
        """Frequently used function to convert info into pandas Series"""
        _zpos = _daxp._FindChannelZpositions(_daxp.xml_filename, verbose=False)[_channel]
        _frames = list(_daxp._FindChannelFrames(_daxp.filename, verbose=False)[_channel])
        _channels = _daxp._FindDaxChannels(_daxp.xml_filename, verbose=False)
        if isinstance(_id, str):
            _bit_name = _id
        else:
            _bit_name = f'bit{_id}'
        # prepare args
        _row = pd.Series([
            _bit_name,
            _readout_name,
            '_'.join(_color_usage_df.get_channel_info_for_round(_hyb)[0]) + f"_s{_daxp.image_size[0]}",
            _file_regExp,
            _bit_num,
            _color_usage_df.get_hyb_id(_hyb),
            _channel,
            '['+' '.join([str(_z) for _z in _frames])+']',
            '['+' '.join([str(_z) for _z in _zpos])+']', 
            '_'.join(_color_usage_df.get_channel_info_for_round(_hyb)[0]) + f"_s{_daxp.image_size[0]}",
            _file_regExp,
            _color_usage_df.get_hyb_id(_hyb),
            ref_Zstep*len(_channels) \
                +_channels.index(_color_usage_df.get_fiducial_channel(_color_usage_df)),
            _color_usage_df.get_fiducial_channel(_color_usage_df),  
        ], index=_columns)
        return _row