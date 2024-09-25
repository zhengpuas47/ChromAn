import os, sys, re, pickle
import numpy as np

## TODO: covert this into class of GeneDict
# class gene_dict

def read_region_file(filename, verbose=True):
    '''Sub-function to read region file'''
    if verbose:
        print (f'Input region file is: {filename}')
    
    # option 1: split txt region files
    if filename.split(os.extsep)[-1] == 'txt':
        with open(filename, 'r') as _reg_file:
            # start reading
            _lines = _reg_file.read().split('\n')
            _titles = _lines[0].split('\t')
            # save a list of dictionaries
            _reg_list = []
            for _line in _lines[1:]:
                _reg_dic = {} # dinctionary to save all informations
                _info = _line.split('\t') # split informations
                if len(_info) != len(_titles): # sanity check to make sure they are of the same size
                    continue
                for _i in range(len(_info)): # save info to dic
                    _reg_dic[_titles[_i]] = _info[_i]
                _reg_list.append(_reg_dic) # save dic to list
    # option 2: split rna txt file
    elif filename.split(os.extsep)[-1] == 'bed':
        _reg_list = []
        with open(filename, 'r') as _reg_file:
            # start reading
            _lines = _reg_file.read().split('\n')
            for _line in _lines:
                _info = _line.split('\t') # split informations
                if len(_info) < 4:
                    continue 
                
                # directly parse
                _dict = {'Chr': _info[0],
                         'Start': _info[1],
                         'End': _info[2],
                         'Name': _info[3].replace('_','-'),
                         } 
                if len(_info) >= 5:
                    _dict['Score'] = _info[4]      
                if len(_info) >= 6:
                    _dict['Strand'] = _info[5]      
                # try to match Txt version:
                if 'chr' in _info[0]:
                    _cname = _info[0].split('chr')[1]
                else:
                    _cname = _info[0]
                _dict.update(
                    {
                        'Gene': _dict['Name'],
                        'Region': f"{_cname}:{_dict['Start']}-{_dict['End']}",
                    }
                )
                _reg_list.append(_dict)

    else:
        raise IOError(f"input file type not supported.")

    if verbose:
        print(f"- {len(_reg_list)} regions loaded from file: {os.path.basename(filename)}")

    return _reg_list

def parse_region(reg_dic):
    '''given a dictionary of one region, 
    report:
        _chrom: str
        _start: int
        _stop: int'''
    region_str = reg_dic['Region']
    # grab chromosome
    _chrom = region_str.split(':')[0]
    _locus = region_str.split(':')[1]
    # grab start and stop positions
    _start, _stop = _locus.split('-')
    _start = int(_start.replace(',', ''))
    _stop = int(_stop.replace(',', ''))
    # return in this order:
    return _chrom, _start, _stop

def gene_dict_2_reg_dict(_gene_dict):
    """Convert standard output TSS into standard region dict"""
    _reg_dict = {
        'Chr':_gene_dict['seqid'],
        'Start':_gene_dict['start'],
        'End':_gene_dict['end'],
        'Name':f"{_gene_dict['infos']['ID']}-{_gene_dict['infos']['Name']}",
        'Gene':_gene_dict['infos']['Name'],
        'Region':f"{_gene_dict['seqid']}:{_gene_dict['start']}-{_gene_dict['end']}",
        'Strand':_gene_dict['strand'],
    }
    return _reg_dict

def reg_dict_2_tss_dict(reg_dict, reg_size):
    """Convert a standard region_dict into TSS and its flanking dict"""
    # initialize
    _tss_dict = {_k:_v for _k,_v in reg_dict.items()}
    # modify start, end
    if _tss_dict['Strand'] == '+': 
        _reg_center = int(reg_dict['Start'])
    elif _tss_dict['Strand'] == '-': 
        _reg_center = int(reg_dict['End'])
    else:
        raise ValueError(f"Wrong input strand, should be + or -")
    # modify tss 
    _tss_dict["Start"] = _reg_center - int(reg_size/2)
    _tss_dict["End"] = _reg_center + int(reg_size/2)
    _tss_dict["Region"] = f"{reg_dict['Chr']}:{_tss_dict['Start']}-{_tss_dict['End']}"
    # modify names
    _tss_dict["Name"] = _tss_dict["Name"] + f"-TSS-{reg_size}" 
    return _tss_dict


class reference_reader:
    """Basic class to reference files"""
    def __init__(self, filename, save=False, save_filename=None, verbose=True):
        # inherit from superclass
        super().__init__()

        # store the filename
        self.ref_filename = filename
        # type
        self.ref_type = filename.split(os.extsep)[-1]
        # file_pointer
        self.fp = None
        # save
        self.save = save
        # print meessage
        self.verbose = verbose
        # determine save_filename
        if save_filename is None:
            self.save_filename = self.ref_filename.replace(self.ref_filename.split(os.extsep)[-1], 'pkl')
        elif isinstance(save_filename, str):
            self.save_filename = save_filename
        else:
            raise TypeError(f"Wrong input type for save_filename, should be string of a filename")
        
    def __str__(self, infotype='all'):
        _str = ''
        if infotype is 'input' or infotype is 'all':
            _str += f"reference file: {self.ref_filename}\n"
            _str += f"reference type: {self.ref_type}\n"
        return _str
    
    def __enter__(self):
        print(f"opening ref_file: {self.ref_filename}")
        self.fp = open(self.ref_filename, "r")
        return self
    
    def __exit__(self, etype, value, traceback):
        if self.fp:
            self.fp.close()
    
    def _load_all(self):
        self.info_lines = [_line.rstrip() for _line in self.fp.readlines()]
        return self.info_lines

    def _load_from_file(self, overwrite=True):
        if os.path.isfile(self.save_filename):
            if self.verbose:
                print(f"- loading from save_file: {self.save_filename}")
            attr_dict = pickle.load(open(self.save_filename, 'rb'))
            for _k, _v in attr_dict.items():
                if not hasattr(self, _k) or overwrite:
                    setattr(self, _k, _v)
        else:
            attr_dict = {}
            if self.verbose:
                print(f"- Invalid save_filename, skip loading.")
        
        return attr_dict
    
    def _save_to_file(self, overwrite=False):
        if os.path.isfile(self.save_filename) and not overwrite:
            if self.verbose:
                print(f"- save_filename:{self.save_filename}, skip saving.")
            attr_dict = {}
        else:
            if self.verbose:
                print(f"- saving to file: {self.save_filename}.")
            attr_dict = {_k: getattr(self, _k) for _k in dir(self) if _k[0] != '_' and _k != 'fp'}
            for _k in attr_dict:
                print(_k, type(_k))
            pickle.dump(attr_dict, open(self.save_filename, 'wb'))
        
        return attr_dict


class gff3_reader(reference_reader):
    """Class to load and parse gff3 reference file (from ensembl, for example)"""
    def __init__(self, filename, save=False, save_filename=None, 
                 load_savefile=True, auto_read=False, verbose=True):
        # inherit from superclass: reference_reader
        super(gff3_reader, self).__init__(filename, save=save, save_filename=save_filename, verbose=verbose)
        # add unique attr
        self.maintext_pt = 0
        # all fields
        self.field_names = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
        # rinitialzie inputs
        self.gene_info_dict = {}
        # do automatic loading from savefile
        if load_savefile:
            self._load_from_file()
        # do automatic reading if specified
        if auto_read:
            # load header and go to the start of main text
            self._load_headers()
            # parse all gene
            self._batch_parse_gene_info()
            # save
            if self.save:
                self._save_to_file()


    def _load_headers(self):
        if self.fp is None:
            self.fp = open(self.ref_filename, "r")
        # if already exist, initialize fp
        else:
            self.fp.seek(0)
        # search header
        _header_lines = []
        # save furrent point
        _pt = self.fp.tell()
        _line = self.fp.readline().rstrip()
        
        while _line != '':
            if len(_line) <= 2:
                continue
            # load an exta line of seps
            elif len(_line) >= 3 and _line[:3] == '###':
                _pt = self.fp.tell()
                self.fp.seek(_pt)
                break
            elif _line[:2] == '##':
                _header_lines.append(_line)
                # add info to existing dicts
                _infos = re.split(r"\s+", _line.split("##")[1])
                if _infos[0] == 'gff-version':
                    self.version = float(_infos[1])
                elif _infos[0] == 'sequence-region':
                    if not hasattr(self, 'sequence_region'):
                        setattr(self, 'sequence_region', {})
                    else:
                        self.sequence_region[_infos[1]] = _infos[2:]
                else:
                    print("header type not supported, skip.")
            elif _line[:2] == '#!':
                _header_lines.append(_line)
                # add info to existing dicts
                _infos = re.split(r"\s+", _line.split("#!")[1])
                if len(_infos) >= 2:
                    setattr(self, _infos[0].replace('-','_'), _infos[1])

            # stop if it's not header revert back
            else:
                self.fp.seek(_pt)
                break
            # update pt
            _pt = self.fp.tell()
            # update line
            _line = self.fp.readline().rstrip()
        
    
        # append attributes
        self.header_lines = _header_lines
        self.maintext_pt = _pt
    
    def _load_gene_by_id(self, gene_id):
        """Load gene_dict information by ENS IDs """
        # load all gene information
        _curr_pt = self.fp.tell()
        # reset pointer if needed
        if _curr_pt > self.maintext_pt:
            if self.verbose:
                print("-- reset pointer back to file start")
            self.fp.seek(self.maintext_pt)
            
        _line = '###'
        _section_lines = []
        _found_gene_flag = False
        while _line != '':
            if len(_line) < 3:
                pass
            elif _line[:3] == '###':
                if _found_gene_flag:
                    break
                else:
                    # empty the buffer
                    _section_lines = []
            else:
                if gene_id in _line:
                    _found_gene_flag = True
                if _found_gene_flag:
                    _section_lines.append(_line)
                    #_infos = re.split("\t+", _line)
                    #print(_infos)
            # update line
            _line = self.fp.readline().rstrip()
            
        return _section_lines
    
    def _parse_gene_info(self, gene_lines, store_result=True):
        _gene_infos = []
        _transcript_infos = {}
        for _line in gene_lines:
            _infos = re.split("\t+", _line)
            # parse
            _dict = {_h:_info for _h, _info in zip(self.field_names, _infos)}
            # attributes
            _dict['infos'] = {}
            for _attr in _dict['attributes'].split(';'):
                _attr_infos = _attr.split('=')
                _dict['infos'][_attr_infos[0]] = _attr_infos[1]
            # childeren
            _dict['Children'] = []
            # update search_list
            _search_dicts = []
            _children_dicts = [_d for _d in _gene_infos]
            while len(_children_dicts) > 0:
                # append
                for _d in _children_dicts:
                    _search_dicts.append(_d)
                # find their childeren
                _grand_children_dicts = []
                for _d in _children_dicts:
                    _grand_children_dicts.extend(_d['Children'])
                _children_dicts = _grand_children_dicts
                
            if 'Parent' not in _dict['infos']:
                _gene_infos.append(_dict)
            else:
                # find parent
                _target_dict = [_pdict for _pdict in _search_dicts 
                                if 'ID' in _pdict['infos'] and _pdict['infos']['ID']==_dict['infos']['Parent']]
                if len(_target_dict) > 0:
                    _target_dict = _target_dict[0]
                    _target_dict['Children'].append(_dict)       

        return _gene_infos
            
    def _batch_parse_gene_info(self):
        """parse the entire gff3 file and process into gene_dicts"""
        if self.verbose:
            print(f"parsing all gene information")
        ## step1: load all
        # check pointer position
        _curr_pt = self.fp.tell()
        # reset pointer if needed
        if _curr_pt > self.maintext_pt:
            if self.verbose:
                print("-- reset pointer back to file start")
            self.fp.seek(self.maintext_pt)
        # load all
        _all_sections = {}
        _gene_name = ''
        _gene_section = []
        for _line in self.fp.readlines():
            _line = _line.rstrip()
            if len(_line) < 3:
                pass
            elif _line[:3] == '###':
                if len(_gene_section) > 0:
                    _all_sections[_gene_name] = _gene_section
                _gene_section = []
                _gene_name = ''
            else:
                if 'gene' in _line:
                    _infos = re.split("\t+", _line)
                    # parse
                    _dict = {_h:_info for _h, _info in zip(self.field_names, _infos)}
                    # attributes
                    _dict['infos'] = {}
                    for _attr in _dict['attributes'].split(';'):
                        _attr_infos = _attr.split('=')
                        _dict['infos'][_attr_infos[0]] = _attr_infos[1]                    
                    if 'ID' in _dict['infos'] and 'Parent' not in _dict['infos']:
                        _gene_name = _dict['infos']['ID']
                        print(_gene_name)
                _gene_section.append(_line)
        # append
        self.gene_lines = _all_sections
        
        # parse all
        if not hasattr(self, 'gene_info_dict'):
            setattr(self, 'gene_info_dict', {})
        for _gene_name, _gene_lines in _all_sections.items():
            _gene_info = self._parse_gene_info(_gene_lines)
            self.gene_info_dict[_gene_name] = _gene_info

        return self.gene_info_dict
    
    ## Searching
    def _search_gene_by_name(self, gene_name):
        """Function to search corresponding gene_information from processed gene_info_dict by gene name
        return all matched genes"""
        
        matched_gene_infos = []
        for _k, _v in self.gene_info_dict.items():
            # search gene_dict within each section
            for _gd in _v:
                if 'ID' in _gd['infos'] and _gd['infos']['ID'] == _k:
                    if 'Name' in _gd['infos'] and _gd['infos']['Name'] == gene_name:
                        matched_gene_infos.append(_gd)
                        
        return matched_gene_infos
    
    def _search_gene_by_id(self, gene_id):
        """Function to search by  corresponding gene_information from processed gene_info_dict
        return all matched genes"""
        matched_gene_infos = []
        # clean up gene_id:
        if ':' in gene_id:
            _id = gene_id.split(':')[-1]
        else:
            _id = gene_id
        
        for _k in self.gene_info_dict:
            if _k.split(':')[-1] == _id:
                for _gd in self.gene_info_dict[_k]:
                    if 'ID' in _gd['infos'] and _gd['infos']['ID'] == _k:
                        matched_gene_infos.append(_gd)

        return matched_gene_infos

