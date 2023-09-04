import sys, os
# required to load parent
from analysis_input import analysis_input_parser,load_analyis_parameters

class AnalysisTask():
    def __init__(self):
        _parser = analysis_input_parser()
        _args, _ = _parser.parse_known_args()
        # assign args        
        for _arg_name in _args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, getattr(_args, _arg_name))
    
    def _load_analysis_parameters(self):
        self.analysis_parameters = load_analyis_parameters(self.analysis_parameters)