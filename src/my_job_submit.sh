#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12gb
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=4         # Enter number of cores/threads you wish to request
#SBATCH --job-name=ChromAn_image_preprocess
#SBATCH --time=2:00:00
#SBATCH --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out
#SBATCH --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err
#SBATCH --partition=weissman         # partition (queue) to use
#SBATCH --account=weissman        # weissman account needed for sabre access
python ./analysis/image_preprocess.py -a test.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20230828-ingel_test_GuHCl-4T1-v21x-0813/Glyoxal_GuHCl_PuWash/Analysis/color_usage.csv -f /lab/weissman_imaging/puzheng/PE_LT/20230828-ingel_test_GuHCl-4T1-v21x-0813/Glyoxal_GuHCl_PuWash/H0M1/Conv_zscan_001.dax
