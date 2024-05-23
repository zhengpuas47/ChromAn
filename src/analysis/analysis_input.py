# %% Function to parse input arguments
import argparse
import pickle
import os
def analysis_input_parser():
    parser = argparse.ArgumentParser(description='ChromAn analysis process')
    parser.add_argument('-f', '--field-of-view', type=int, # No default, this has to be specified
        help='the field-of-view id to be process; required')
    parser.add_argument('-d', '--data-folder', type=str, # No default, this has to be specified
        help='the data-folder to be process; required')
    parser.add_argument('-c', '--color-usage', type=str, default='color_usage.csv',
                        help='name of the color_usage file to use')
    parser.add_argument('-i', '--hyb-id', type=int, 
                        default=-1,
                        help='the index of the hybridization round to analyze, -1 for all fovs')
    parser.add_argument('-a', '--analysis-parameters', type=str, default='parameters.pkl',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-t', '--test', type=bool, default=False,
                        help='Whether to run the test')

    return parser

def load_analyis_parameters(analysis_parameters):
    if os.path.isfile(analysis_parameters):
        # pickle
        parameter_dict = pickle.load(open(analysis_parameters, 'rb'))
    else:
        print("analysis_parameters not found, use default parameters")
        parameter_dict = dict()
    # cleanup formatting:
    ## TODO: fill in more details to refine parameter
    # return
    return parameter_dict

# %%
def organization_input_parser():
    parser = argparse.ArgumentParser(description='ChromAn analysis process')
    parser.add_argument('-c', '--color-usage', type=str, default='color_usage.csv',
                        help='name of the color_usage file to use')
    parser.add_argument('-s', '--save-folder', type=str, default=None,
                        help='name of the save folder')
    parser.add_argument('-r', '--correction-folder', type=str, default='/lab/weissman_imaging/puzheng/Corrections/20231012-Merscope01_s40_n500',
                        help='Whether to run the test')    
    parser.add_argument('-o', '--overwrite', type=bool, default=False,
                        help='Whether to overwrite the existing file')
    parser.add_argument('-t', '--test', type=bool, default=False,
                        help='Whether to run the test')

    return parser
