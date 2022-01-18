#! /usr/bin/env python
#
# Copyright (C) 

from distutils.core import setup
from distutils import sysconfig
from setuptools import setup, Extension, Command
from setuptools import setup, find_packages
import os,sys
from shutil import copy
__version__ = '0.0.1'


DISTNAME = 'codpy'
DESCRIPTION = 'An RKHS based module for machine learning and data mining'
#with open('README.rst') as f:
  #  LONG_DESCRIPTION = f.read()
MAINTAINER = 'jean-marc mercier'
MAINTAINER_EMAIL = 'jeanmarc.mercier@gmail.com'
URL = 'https://github.com/johnlem/codpy/dist'
#DOWNLOAD_URL = 'https://github.com/johnlem/codpy/dist'
LICENSE = 'new BSD'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/ ..../issues',
    'Documentation': 'https://',
    'Source Code': 'https://github.com/.....'
}


#print("find_packages():",find_packages(),)

setup(
    name='codpy',
    version=__version__,
    author=MAINTAINER,
    maintainer=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    packages=['codpy'],
    include_package_data=True,
    classifiers=[
        # trove classifiers
        # the full list is here: https://pypi.python.org/pypi?%3aaction=list_classifiers
        'development status :: 2 - pre-alpha',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering'
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Microsoft :: Windows',
        
    ],
    install_requires=['pybind11','pandas>=1.0','numpy>=1.18','matplotlib>=3.2','mkl','scikit-learn>=0.24.1','scipy>=1.6.1',
    'tensorflow','seaborn', 'scikit-image','tensorflow-datasets','torch','xgboost','jupyter','quantlib','xlrd','pydicom'],
    extras_require={
    'win32': 'pywin32'
  }
)
