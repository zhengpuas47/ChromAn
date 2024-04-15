import os, sys
import scanpy as sc

sys.path.append('..')


# Class to contain the gene selection
class GeneSelection:
    def __init__(self, adata_filename):
        self.adata_filename = adata_filename
        self.adata = sc.read_h5ad(adata_filename)
        self.gene_id = None
        self.gene_name = None
        
    def __str__(self):
        return f"GeneSelection: {self.gene_id} {self.gene_name}"

## TODO: update gene selection process here