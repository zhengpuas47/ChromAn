#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=1         # Enter number of cores/threads you wish to request
#SBATCH --job-name=ChromAn_image_preprocess
#SBATCH --time=3:00:00
#SBATCH --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out
#SBATCH --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err
#SBATCH --partition=weissman # partition (queue) to use
#SBATCH --account=weissman # weissman account needed for sabre access
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_00.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_01.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_02.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_03.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_04.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_05.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_06.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_07.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_08.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_09.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_10.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_11.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_12.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_13.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_14.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_15.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_16.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_17.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_18.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_19.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_20.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_21.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_22.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_23.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_24.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_25.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_26.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_27.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_28.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_29.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_30.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_31.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_32.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_33.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_34.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_35.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_36.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_37.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_38.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_39.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_40.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_41.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_42.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_43.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_44.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_45.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_46.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_47.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_48.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_49.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_50.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_51.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_52.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_53.dax" 
sbatch --nodes=1 --ntasks=1 --open-mode=append --cpus-per-task=1 --job-name=ChromAn_image_preprocess --mem=20gb --time=3:00:00 --output=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.out --error=/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs/%x_%j.err --partition=weissman --account=weissman --wrap="python ./analysis/image_preprocess.py -a /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/Analysis/H0_preprocessParam.pkl -c /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/Analysis/color_usage_fixation.csv -f /lab/weissman_imaging/puzheng/PE_LT/20231010-4T1v21x-50k0920_fixation_test/MeAA/H0M1/Conv_zscan_54.dax" 
