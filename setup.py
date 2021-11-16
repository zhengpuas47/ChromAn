import os
from setuptools import setup


install_requires = [line.rstrip() for line in open(
    os.path.join(os.path.dirname(__file__), "package_requirements.txt"))]

setup(
    name='ChromAn',
    version='0.1a0',
    description="Chromatin imaging analysis software",
    author="Pu Zheng",
    author_email="zhengpuas47@gmail.com",
    url='https://github.com/zhengpuas47/ChromAn',
    license="GPL 3.0",
    packages=['ChromAn'],
    package_dir = {'': 'src'},
    install_requires=install_requires, 
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
