# Packages
import os, sys, time, json
import numpy as np
import json
from datetime import datetime
# local functions and variables
sys.path.append('..')
from default_parameters import default_slurm_output,default_slurm_prameters,default_analysis_home
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
    parser.add_argument('-m', '--memory', type=str,
                        default='20gb',
                        help='total memory usage')
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

def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    elif isinstance(stringIn, str):
        return stringIn.strip('\'').strip('\"').rstrip("/")
    else:
        return stringIn
    
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
#SBATCH --mem={self.slurm_parameters['mem']}
#SBATCH --output={self.slurm_parameters['output']}
#SBATCH --partition={self.slurm_parameters['partition']} # partition (queue) to use
#SBATCH --account={self.slurm_parameters['account']} # account to use
"""     
        slurm_script += """# Run the python script in parallel \n""" # add comments
        # loop through images
        for _fid in self.fov_ids:
            # print info:
            if self.verbose:
                print(f"Processing {self.task_name} for FOV: {_fid}")
            # generate command
            command = f"""srun --ntasks={1} --job-name={self.slurm_parameters['job-name']}_{self.task_name}_{_fid} """
            # GPU related settings:
            if self.slurm_parameters['use_gpu']:
                command += f"""--gres=gpu:1 \
--partition={self.slurm_parameters['gpu_partition']} --account={self.slurm_parameters['gpu_account']} """
            else:
                command += f"""--partition={self.slurm_parameters['partition']} --account={self.slurm_parameters['account']} """
            # append run-task
            command += f"""python {os.path.join(default_analysis_home,'analysis',self.task_name+'.py')} \
-d {self.data_folder} -a {self.analysis_parameters} -c {self.color_usage} -f {_fid} """
            # append this line
            slurm_script += command + "& \n"
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


class GenerateAnalysisTask(object):

    def __init__(self, 
                 tasks_per_node=1, 
                 time_limit='3:00:00'):
        # parse analysis 
        parser = build_parser()
        args, argv = parser.parse_known_args()
        # assign args        
        for _arg_name in args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, getattr(args, _arg_name))
        self.tasks_per_node = tasks_per_node
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
        _cpu_partition = 'weissman' # 20
        _cpu_account = 'weissman' # 'wibrusers' 
        _gpu_partition = 'nvidia-2080ti-20' # 'sabre' 
        _gpu_account = 'wibrusers' # 'weissman' 
        # GPU related settings:
        if self.use_gpu:
            print("Use GPU")
            _partiton, _account = _gpu_partition, _gpu_account
            #slurm_header += """#SBATCH --gres=gpu:1              # This is needed to actually access a gpu\n"""
        else:
            _partiton, _account = _cpu_partition, _cpu_account
        # for the header, always use cpu partition to submit jobs
        slurm_header += \
f"""#SBATCH --partition={_cpu_partition} # partition (queue) to use
#SBATCH --account={_cpu_account} # weissman account needed for sabre access
"""
        # generate command
        full_script = slurm_header
        for _filename in self.image_filenames:
            command = f"""sbatch \
--nodes={1} --ntasks={1} --open-mode=append \
--cpus-per-task={self.core_count} --job-name=ChromAn_{self.analysis_task} \
--mem={self.memory} \
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
#if __name__ == "__main__":
#    task = GenerateAnalysisTask()
    # generate slurm
#    task.generate_slurm_script()
