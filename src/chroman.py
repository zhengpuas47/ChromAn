# Packages
import os, sys, time, json
import numpy as np
import json
from datetime import datetime
# local functions and variables
sys.path.append('..')
from default_parameters import default_slurm_output,default_slurm_prameters,default_analysis_home, default_data_folder_regexp, default_data_fov_regexp
# create
from file_io.data_organization import search_fovs_in_folders, Color_Usage

import argparse
# clean string
def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    elif isinstance(stringIn, str):
        return stringIn.strip('\'').strip('\"').rstrip("/")
    else:
        return stringIn
# input parser   
def chroman_input_parser():
    """Parse the input arguments for chroman.py.
    """
    parser = argparse.ArgumentParser(description='Chromatin analysis')
    # define arguments
    parser.add_argument('-t', '--task-name', type=str,
                        help='the name of the task to execute')
    parser.add_argument('-d', '--data-folder', type=str,
                        help='directory where the raw data is stored')
    parser.add_argument('-c', '--color-usage', type=str,
                        default='color_usage.csv',
                        help='name of the color-usage file to use, in format of x-y or int, none means all fovs')    
    parser.add_argument('-f', '--field-of-view', type=str,
                        default='all',
                        help='the index of the field of view to analyze')
    parser.add_argument('-i', '--hyb-id', type=int,
                        default=-1,
                        help='the index of the hybridization round to analyze, -1 for all fovs')
    # parse args for folder and fovs
    parser.add_argument('-p', '--data-folder-regexp', type=str,
                        default=default_data_folder_regexp,
                        help='regular expression to find folders')
    parser.add_argument('-q', '--data-fov-regexp', type=str,
                        default=default_data_fov_regexp,
                        help='regular expression to find each fovs')
    parser.add_argument('-a', '--analysis-parameters',
                        default='analysis_parameters.json',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-s', '--slurm-parameters', type=str,
                        default='slurm_parameters.json',
                        help='parameters for SLURM script')
    parser.add_argument('-o', '--save-folder', type=str,
                        default='.',
                        help='directory where the SLURM script will be saved')
    parser.add_argument('-v', '--verbose', 
                    default=False, action="store_true",
                    help='whether print details',)
    return parser

class ChromAn_slurm(object):
    """Generate SLURM script for Chromatin analysis.
    by Pu Zheng, 2024
    From the given inputs and parameters,
    Require at least two inputs: 
    1. task_name
    2. data_folder
    3. color_usage
    Optional:
    3. field_of_view
    4. analysis_parameters,
    5. slurm_parameters
    6. save_folder
    7. verbose
    """
    def __init__(self) -> None:
        # parse input
        parser = chroman_input_parser()
        args, argv = parser.parse_known_args()
        # assign args
        for _arg_name in args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, _clean_string_arg(getattr(args, _arg_name)))
        # load slurm parameters
        if os.path.isfile(self.slurm_parameters):
            self.slurm_parameters = json.load(open(self.slurm_parameters, 'r'))
        else:
            Warning(f"Slurm parameters file doesn't exist in {self.slurm_parameters}, use default parameters")
            self.slurm_parameters = {}
        # use default parameters if not specified
        for _k, _v in default_slurm_prameters.items():
            if _k not in self.slurm_parameters:
                self.slurm_parameters[_k] = _v

    def generate_slurm_script(self):
        """Generate SLURM script with the given command."""
        # scan subfolders
        folders, fovs = search_fovs_in_folders(self.data_folder, verbose=self.verbose)        
        # load color_usage
        color_usage_full_filename = os.path.join(self.data_folder, 'Analysis', self.color_usage)
        if os.path.isfile(color_usage_full_filename):
            color_usage_df = Color_Usage(color_usage_full_filename)
        else:
            raise FileExistsError(f"Color usage file doesn't exist in {color_usage_full_filename}, exit")
        # decide fovs
        if isinstance(self.field_of_view, str) and '-' in self.field_of_view: # format 1
            _start, _end = self.field_of_view.split('-')
            self.fov_ids = np.arange(int(_start), min(int(_end)+1, len(fovs)))
        elif isinstance(self.field_of_view, str) and self.field_of_view.isdigit(): # format 2 direct integer
            self.fov_ids = [int(self.field_of_view)]
        else: # not specified
            self.fov_ids = np.arange(len(fovs))
        # create the slurm file:
        slurm_script = f"""#!/bin/bash
#SBATCH --nodes={self.slurm_parameters['nodes']}
#SBATCH --ntasks={len(self.fov_ids)}
#SBATCH --cpus-per-task={self.slurm_parameters['cpus-per-task']}         # Enter number of cores/threads you wish to request
#SBATCH --job-name=ChromAn_{self.task_name}
#SBATCH --time={self.slurm_parameters['time']}
#SBATCH --output={self.slurm_parameters['output']}
#SBATCH --partition={self.slurm_parameters['partition']} # partition (queue) to use
#SBATCH --account={self.slurm_parameters['account']} # account to use
#SBATCH --mem={self.slurm_parameters['mem']}
"""     

        slurm_script += """# Run the python script in parallel \n""" # add comments
        # loop through images
        for _fid in self.fov_ids:
            # print info:
            if self.verbose:
                print(f"Processing {self.task_name} for FOV: {_fid}")
            # generate command
            command = f"""sbatch --ntasks={1} --job-name={self.slurm_parameters['job-name']}_{self.task_name}_{_fid} """
            # GPU related settings:
            if self.slurm_parameters['use_gpu']:
                command += f"""--gres=gpu:1 --mem={self.slurm_parameters['mem']} \
--output={self.slurm_parameters['output']} --error={self.slurm_parameters['error']} \
--partition={self.slurm_parameters['gpu_partition']} --account={self.slurm_parameters['gpu_account']} """
            # CPU related settings:
            else:
                command += f"""--mem={self.slurm_parameters['mem']} \
--output={self.slurm_parameters['output']} --error={self.slurm_parameters['error']} \
--partition={self.slurm_parameters['partition']} --account={self.slurm_parameters['account']} """
            # append run-task
            command += f"""--wrap="python {os.path.join(default_analysis_home,'analysis',self.task_name+'.py')} \
-d {self.data_folder} -i {self.hyb_id} -a {self.analysis_parameters} -c {self.color_usage} -f {_fid}" """
            # append this line
            slurm_script += command + "\n"
        # add wait step
        slurm_script += """# Wait for all abckground jobs to finish \n""" # add comments
        slurm_script += "wait\n"
        # save the slurm output:
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            if self.verbose:
                print(f"Create folder {self.save_folder}")
        # savefile:
        script_filename = os.path.join(self.save_folder, 
            f"{os.path.basename(self.data_folder)}_{self.task_name}_fov-{self.field_of_view}_{datetime.now().strftime('%Y-%m-%d')}.sh")
        with open(script_filename, 'w') as file:
            file.write(slurm_script)
        print(f"SLURM script saved to {script_filename}")
        return 
            

# Example usage:
if __name__ == "__main__":
    chroman_slurm = ChromAn_slurm()
    # generate slurm
    chroman_slurm.generate_slurm_script()
