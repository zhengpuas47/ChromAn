import numpy as np 
import os, sys
import re

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

