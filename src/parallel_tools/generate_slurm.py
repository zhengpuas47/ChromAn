# Packages
import os, sys, time, json
import numpy as np
# local functions and variables
sys.path.append('../..')
from src.default_parameters import default_slurm_output
# create


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

class AnalysisTask(object):

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

    def generate_slurm_script(self, script_filename='submit.sh'):
        """Generate SLURM script with the given command."""
        
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
        command = f"python ../analysis/{self.analysis_task}.py -a {self.analysis_parameters} -i {self.fragment_index} -d {self.dataset}"
        full_script = slurm_header + command + "\n"

        with open(script_filename, 'w') as file:
            file.write(full_script)

        print(f"SLURM script saved to {script_filename}")


# Example usage:
if __name__ == "__main__":
    
    #print(args.analysis_task)
    task = AnalysisTask()
    
    #task = AnalysisTask('image_preprocess', nodes=2, tasks_per_node=4, memory='2G', job_name='analysis_task')
    task.generate_slurm_script(script_filename="my_job_submit.sh")
