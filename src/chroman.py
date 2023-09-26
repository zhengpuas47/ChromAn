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
    # arguments
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
                        default=1,
                        help='number of CPU cores to use for each task')
    parser.add_argument('--nodes', type=int,
                        default=1,
                        help='number of cluster nodes to allocate jobs')
    parser.add_argument('-o', '--job-output', type=str,
                        default=default_slurm_output, 
                        help='slurm output directory',)
    parser.add_argument('-s', '--script', type=str, 
                        help="directory of output script file", )
    # boolean
    parser.add_argument('-g', '--use-gpu', 
                        default=False, action="store_true",
                        help='whether submit job to GPU node',)    
    parser.add_argument('-v', '--verbose', 
                        default=False, action="store_true",
                        help='whether print details',)
    # data_folder    
    parser.add_argument('data_folder', type=str,
                        help='directory where the raw data is stored')
    
    
    return parser



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
                 tasks_per_node=1, 
                 memory='12gb', time_limit='3:00:00'):
        # parse analysis 
        parser = build_parser()
        args, argv = parser.parse_known_args()
        # assign args        
        for _arg_name in args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, getattr(args, _arg_name))
        self.tasks_per_node = tasks_per_node
        self.memory = memory
        self.time_limit = time_limit
        # default shell script name:
        if not hasattr(self, 'script') or self.script is None:
            self.script = f"{os.path.basename(self.data_folder.strip('/'))}_{self.analysis_task}.sh"

    def identify_image_filenames(self, ):
        # scan subfolders
        folders, fovs = search_fovs_in_folders(task.data_folder)
        # load color_usage
        color_usage_full_filename = os.path.join(self.data_folder, 'Analysis', self.color_usage)
        self.color_usage = color_usage_full_filename
        if os.path.isfile(color_usage_full_filename):
            color_usage_df = Color_Usage(color_usage_full_filename)
        else:
            raise FileExistsError(f"Color usage file doesn't exist in {color_usage_full_filename}, exit")

        # identify valid folders
        #command_list = []
        self.image_filenames = []
        #f"python ./analysis/{self.analysis_task}.py -a {self.analysis_parameters} -c {self.color_usage} -f {self.image_filename}"
        # hyb folders:
        if not hasattr(self, 'hyb_folder'):
            self.folders = color_usage_df.index
        elif self.hyb_folder in color_usage_df.index:
            self.folders = [self.hyb_folder]
        else:
            raise ValueError("Invalid hyb_folder")
        # fovs:
        if not hasattr(self, 'fragment_index') or self.fragment_index is None:
            self.fov_ids = np.arange(len(fovs))
        elif self.fragment_index >= 0 and self.fragment_index < len(fovs):
            self.fov_ids = [self.fragment_index]
        else:
            raise ValueError("Invalid fragment_index")
        # loop through images
        for _fd in self.folders:
            for _fov_id in self.fov_ids:
                self.image_filenames.append(os.path.join(self.data_folder, _fd, fovs[_fov_id]))
        print(f"{len(self.image_filenames)} images in {len(self.folders)} folders are going to be proecessed. ")
    
    def generate_slurm_script(self):
        """Generate SLURM script with the given command."""
        # identify image filename
        if not hasattr(self, 'image_filename'):
            self.identify_image_filenames()
## The following comments are removed to properly excecute on slurm:
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={len(self.image_filenames)}        
        slurm_header = f"""#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --cpus-per-task={self.core_count}         # Enter number of cores/threads you wish to request
#SBATCH --job-name=ChromAn_{self.analysis_task}
#SBATCH --time={self.time_limit}
#SBATCH --output={os.path.join(self.job_output, r'%x_%j.out')}
#SBATCH --error={os.path.join(self.job_output, r'%x_%j.err')}
"""
        # Set partitions and accounts
        _cpu_partition = 'weissman'
        _cpu_account = 'weissman'
        _gpu_partition = 'nvidia-2080ti-20'
        _gpu_account = 'wibrusers'
        # GPU related settings:
        if self.use_gpu:
            _partiton, _account = _gpu_partition, _gpu_account
            slurm_header += """#SBATCH --gres=gpu:1              # This is needed to actually access a gpu\n"""
        else:
            _partiton, _account = _cpu_partition, _cpu_account
            
        slurm_header += \
f"""#SBATCH --partition={_partiton} # partition (queue) to use
#SBATCH --account={_account} # weissman account needed for sabre access
"""
        # generate command
        full_script = slurm_header
        for _filename in self.image_filenames:
            command = f"""sbatch \
--nodes={1} --ntasks={1} --open-mode=append \
--cpus-per-task={self.core_count} --job-name=ChromAn_{self.analysis_task} \
--time={self.time_limit} --output={os.path.join(self.job_output, r'%x_%j.out')} \
--error={os.path.join(self.job_output, r'%x_%j.err')} \
"""         
            # GPU related settings:
            if self.use_gpu:
                command += f"""--gres=gpu:1 """
            # partition and account
            command += f"""--partition={_partiton} --account={_account} """         
            # append the rest of command
            command += f"""--wrap="python ./analysis/{self.analysis_task}.py -a {self.analysis_parameters} -c {self.color_usage} -f {_filename}" \
"""
            #command = f"srun -n1 --mem {self.memory} --exclusive python ./analysis/{self.analysis_task}.py -a {self.analysis_parameters} -c {self.color_usage} -f {_filename}"
            full_script += command + "\n"

        # save
        if hasattr(self, 'script'):
            script_filename = getattr(self, 'script')
        with open(script_filename, 'w') as file:
            file.write(full_script)

        print(f"SLURM script saved to {script_filename}")


# Example usage:
if __name__ == "__main__":
    task = GenerateAnalysisTask()
    # generate slurm
    task.generate_slurm_script()
