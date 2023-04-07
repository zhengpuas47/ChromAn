import os
from setuptools import setup
#import distutils.command.bdist_conda

install_requires = [line.rstrip() for line in open(
    os.path.join(os.path.dirname(__file__), "package_requirements.txt"))]

setup(
    name='ChromAn',
    version='0.1',
    description="Chromatin and MERFISH imaging analysis software",
    author="Pu Zheng",
    author_email="zhengpuas47@gmail.com",
    url='https://github.com/zhengpuas47/ChromAn',
    license="GPL 3.0",
    packages=['chrom_models'],
    package_dir = {'': 'src'},
    #distclass=distutils.command.bdist_conda.CondaDistribution,
    install_requires=install_requires, 
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
