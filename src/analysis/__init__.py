# required to load parent
from analysis_input import analysis_input_parser,load_analyis_parameters

class AnalysisTask():
    def __init__(self):
        _parser = analysis_input_parser()
        _args, _ = _parser.parse_known_args()
        # assign args        
        for _arg_name in _args.__dir__():
            if _arg_name[0] != '_':
                setattr(self, _arg_name, _clean_string_arg(getattr(_args, _arg_name)))
    
    def _load_analysis_parameters(self):
        if self.analysis_parameters is None:
            print("No analysis_parameters provided!")
            self.analysis_parameters = dict()
        self.analysis_parameters = load_analyis_parameters(self.analysis_parameters)
        
        
def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    return stringIn.strip('\'').strip('\"')