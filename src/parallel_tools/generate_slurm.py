# Packages
import os, sys, time, json
import numpy as np
# local functions and variables

# create


class AnalysisTask(object):

    def __init__(self, nodes=1, tasks_per_node=1, memory='10G', job_name='ChromAn_analysis', time_limit='1:00:00'):
        self.nodes = nodes
        self.tasks_per_node = tasks_per_node
        self.memory = memory
        self.job_name = job_name
        self.time_limit = time_limit

    def generate_slurm_script(self, command, script_name='submit.sh'):
        """Generate SLURM script with the given command."""
        
        slurm_header = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.tasks_per_node}
#SBATCH --mem={self.memory}
#SBATCH --job-name={self.job_name}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.job_name}.out
#SBATCH --error={self.job_name}.err

"""

        full_script = slurm_header + command

        with open(script_name, 'w') as file:
            file.write(full_script)

        print(f"SLURM script saved to {script_name}")


# Example usage:
if __name__ == "__main__":
    task = AnalysisTask(nodes=2, tasks_per_node=4, memory='2G', job_name='analysis_job')
    task.generate_slurm_script("srun python my_script.py", script_name="my_job_submit.sh")
