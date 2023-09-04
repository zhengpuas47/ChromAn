# Packages
import os, sys, time, json
import numpy as np
# local functions and variables
sys.path.append('..')
from src.default_parameters import default_slurm_output
# create
from file_io.data_organization import search_fovs_in_folders, Color_Usage

import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='Decode Chromatin data.')
    
    parser.add_argument('-t', '--analysis-task',
        help='the name of the analysis task to execute. If no '
             + 'analysis task is provided, all tasks are executed.') # for now you need to specify a task
    parser.add_argument('-f', '--hyb-folder', type=str,
        help='the sub-folder name of the analysis task to execute')
    parser.add_argument('-i', '--fragment-index', type=int,
        help='the index of the fragment of the analysis task to execute')
    parser.add_argument('-a', '--analysis-parameters',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-c', '--color-usage', type=str,
                        default='color_usage.csv',
                        help='name of the color-usage file to use')
    parser.add_argument('-n', '--core-count', type=int,
                        default=4,
                        help='number of CPU cores to use for the analysis')
    parser.add_argument('-o', '--job-output', type=str,
                        default=default_slurm_output, 
                        help='slurm output directory',)
    parser.add_argument('-g', '--use-gpu', type=bool,
                        default=False, 
                        help='whether submit job to GPU node',)
    parser.add_argument('dataset',
                        help='directory where the raw data is stored')
    
    return parser

def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    return stringIn.strip('\'').strip('\"')

def _get_input_path(prompt):
    while True:
        pathString = str(input(prompt))
        if not pathString.startswith('s3://') \
                and not os.path.exists(os.path.expanduser(pathString)):
            print('Directory %s does not exist. Please enter a valid path.'
                  % pathString)
        else:
            return pathString#

class GenerateAnalysisTask(object):

    def __init__(self, 
                 nodes=1, tasks_per_node=1, 
                 memory='12gb', time_limit='2:00:00'):
        # parse analysis 
        parser = build_parser()
        args, argv = parser.parse_known_args()
        print(args.__dir__())
        # assign args        
        for _arg_name in args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, getattr(args, _arg_name))
        #self.analysis_task = args.analysis_task # task name
        #self.analysis_parameters = args.analysis_parameters # parameter filename to use
        #self.fragment_index = args.fragment_index # field of view index
        #self.core_count = args.core_count # number of CPU cores
        #self.job_output = args.job_output
        #self.use_gpu = args.use_gpu # whether use GPU
        #self.dataset = args.dataset
        self.nodes = nodes
        self.tasks_per_node = tasks_per_node
        self.memory = memory
        #self.job_name = job_name
        self.time_limit = time_limit

    def identify_image_filenames(self, ):
        # scan subfolders
        folders, fovs = search_fovs_in_folders(task.dataset)
        # load color_usage
        color_usage_full_filename = os.path.join(self.dataset, 'Analysis', self.color_usage)
        self.color_usage = color_usage_full_filename
        if os.path.isfile(color_usage_full_filename):
            color_usage_df = Color_Usage(color_usage_full_filename)
        else:
            raise FileExistsError("Color usage file doesn't exist in given path, exit")

        # identify valid folders
        if hasattr(self, 'hyb_folder') and self.hyb_folder in color_usage_df.index:
            self.image_filename = os.path.join(self.dataset, self.hyb_folder, fovs[self.fragment_index])
        else:
            raise ValueError("Invalid hyb_folder")
    
    def generate_slurm_script(self, script_filename='submit.sh'):
        """Generate SLURM script with the given command."""
        # identify image filename
        if not hasattr(self, 'image_filename'):
            self.identify_image_filenames()
        
        slurm_header = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.tasks_per_node}
#SBATCH --mem={self.memory}
#SBATCH --open-mode=append
#SBATCH --cpus-per-task={self.core_count}         # Enter number of cores/threads you wish to request
#SBATCH --job-name=ChromAn_{self.analysis_task}
#SBATCH --time={self.time_limit}
#SBATCH --output={os.path.join(self.job_output, r'%x_%j.out')}
#SBATCH --error={os.path.join(self.job_output, r'%x_%j.err')}
"""
        # GPU related settings:
        if self.use_gpu:
            slurm_header += \
"""SBATCH --gres=gpu:1              # This is needed to actually access a gpu
#SBATCH --partition=sabre         # partition (queue) to use
#SBATCH --account=weissman        # weissman account needed for sabre access
"""
        else:
            slurm_header += \
"""#SBATCH --partition=weissman         # partition (queue) to use
#SBATCH --account=weissman        # weissman account needed for sabre access
"""         
        # generate command
        command = f"python ./analysis/{self.analysis_task}.py -a {self.analysis_parameters} -c {self.color_usage} -f {self.image_filename}"
        full_script = slurm_header + command + "\n"

        with open(script_filename, 'w') as file:
            file.write(full_script)

        print(f"SLURM script saved to {script_filename}")


# Example usage:
if __name__ == "__main__":
        
    task = GenerateAnalysisTask()
    # generate slurm
    task.generate_slurm_script(script_filename="my_job_submit.sh")
