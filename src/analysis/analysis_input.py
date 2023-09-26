# %% Function to parse input arguments
import argparse
import pickle
def analysis_input_parser():
    parser = argparse.ArgumentParser(description='ChromAn analysis process')
    parser.add_argument('-f', '--image-filename', type=str,
        help='the image-filename to be process')
    parser.add_argument('-c', '--color-usage', type=str,
                        help='name of the analysis parameters file to use')
    parser.add_argument('-a', '--analysis-parameters', type=str,
                        help='name of the analysis parameters file to use')
    parser.add_argument('-s', '--overwrite', type=bool, default=False,
                        help='name of the analysis parameters file to use')
    
    return parser

def load_analyis_parameters(analysis_parameters):
    # pickle
    parameter_dict = pickle.load(open(analysis_parameters, 'rb'))
    # cleanup formatting:
    ## TODO: fill in more details to refine parameter
    # return
    return parameter_dict

# %%