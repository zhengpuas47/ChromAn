import os, sys, re
import pandas as pd
import numpy as np

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
    def custom_method(self):
        # define custom method here
        pass

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
    def get_fiducial_channel(color_usage_df, fiducial_query='beads'):
        for _c in color_usage_df.columns:
             if fiducial_query in color_usage_df[_c].fillna(-1).values:
                 return _c
        return None
    
        
