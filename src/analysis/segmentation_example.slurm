#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --job-name=cellpose
#SBATCH --mem 24GB
#SBATCH --open-mode=append
#SBATCH --partition=sabre            # partition (queue) to use; 20 for shared, weissman for dedicated
#SBATCH --account=weissman              # this is necessary if partition is weissman, for partition-20, use wibrusers
#SBATCH -o /lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.stdout
#SBATCH -e /lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.stdout

#SBATCH --array=1-651%2
#SBATCH --gres=gpu:1

source /home/puzheng/miniconda3/etc/profile.d/conda.sh
conda activate fishtank

Data='20250430-Hela500k_SC1'

python /lab/weissman_imaging/puzheng/Softwares/ChromAn/src/analysis/cell_segment.py\
 -f $SLURM_ARRAY_TASK_ID\
 -d /lab/weissman_imaging/puzheng/SC/${Data}/\
 -c /lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/color_usage_SC1_v1.csv\
 -a /lab/weissman_imaging/puzheng/SC/20250430-Hela500k_SC1/parameters_v2.pkl\
 -s /lab/weissman_imaging/puzheng/MERFISH_analysis/SC/Postanalysis/${Data}/\
 -r /lab/weissman_imaging/puzheng/Corrections/20240401-Merscope01_s11_n1200\
 -m /lab/weissman_imaging/puzheng/Softwares/Weissman_MERFISH_Scripts/merlin_parameters/microscope/merscope01_microscope.json\
 -t False
