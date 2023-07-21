from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import os
setup(name='Fast oligo tools',ext_modules=cythonize( '.'+os.path.sep+r"seqint.pyx"), include_dirs=[np.get_include()])
#setup(name='Fast oligo tools',ext_modules=cythonize(r"C_Tools\seqint.pyx"), include_dirs=[np.get_include()])
