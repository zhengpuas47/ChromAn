# %% Function to parse input arguments
import argparse
import pickle
import os
def analysis_input_parser():
    parser = argparse.ArgumentParser(description='ChromAn analysis process')
    parser.add_argument('-f', '--image-filename', type=str,
        help='the image-filename to be process; required')
    parser.add_argument('-c', '--color-usage', type=str, default='color_usage.csv',
                        help='name of the color_usage file to use')
    parser.add_argument('-a', '--analysis-parameters', type=str, default='parameters.pkl',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-s', '--overwrite', type=bool, default=False,
                        help='Whether to overwrite the existing file')
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