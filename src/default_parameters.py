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
default_correction_folder = r'../data/example_correction_profile/'
# number of threads
default_num_threads = 12


# default slurm output, specific to WI server
default_slurm_output = r'/lab/weissman_imaging/puzheng/slurm_reports/ChromAn_Jobs'
