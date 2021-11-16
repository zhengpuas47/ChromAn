from setuptools import setup

setup(
    name='ChromAn',
    version='0.1a0',
    packages=['ChromAn'],
    install_requires=[
        'requests',
        'importlib; python_version == "3.8"',
    ],
)
