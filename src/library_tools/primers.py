import pandas as pd
import os

def load_fasta_to_DataFrame(filename):
    """Quick tool to convert a fasta file into dataframe"""
    from Bio.SeqIO.FastaIO import SimpleFastaParser

    _seq_df = {"Name":[], "Sequence":[]}
    with open(filename, 'r') as _handle:
        for _name, _seq in SimpleFastaParser(_handle):
            _seq_df['Name'].append(_name)
            _seq_df['Sequence'].append(_seq)
    return pd.DataFrame(_seq_df)
