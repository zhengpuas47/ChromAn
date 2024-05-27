import numpy as np
# default image parameters
default_im_size=np.array([50,2048,2048])
default_pixel_size=np.array([250,108,108])
default_channels = ['750', '647', '561', '488', '405']
default_corr_channels = ['750', '647', '561']
default_ref_channel = '647'
default_fiducial_channel = '488'
default_dapi_channel = '405'
default_num_buffer_frames = 0
default_num_empty_frames = 0
default_seed_th = 1000

# decoding parameter
default_search_radius = 4
# default directory parameters
default_correction_folder = r'/lab/weissman_imaging/puzheng/Corrections/20230902-Merscope01_s30_n500'
# number of threads
default_num_threads = 12
# default regexp for finding folders and files
default_data_folder_regexp = r'^H([0-9]+)[RQBUGCMP]([0-9]+)(.*)'
default_data_fov_regexp = r'(.+)_([0-9]+)\.dax'

# default slurm output, specific to WI server
default_slurm_output = r'/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs'
default_analysis_home = r'/lab/weissman_imaging/puzheng/Softwares/ChromAn/src'
default_slurm_prameters = {
    'partition': 'weissman',
    'account': 'weissman',
    'gpu_partition': 'sabre',
    'gpu_account': 'weissman',
    'time': '24:00:00',
    'mem': 20000,
    'nodes': 1,
    'ntasks': 1,
    'cpus-per-task': 4,
    'job-name': 'ChromAn',
    'output': default_slurm_output + '/%x-%j.out',
    'error': default_slurm_output + '/%x-%j.err',
    'use_gpu': False,
}