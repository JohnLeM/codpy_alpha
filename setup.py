#! /usr/bin/env python
#
# Copyright (C) 

from setuptools import setup, Extension, Command
from setuptools import setup, find_packages
from distutils.core import setup
from distutils import sysconfig
import os,sys
from shutil import copy
__version__ = '0.0.1'

codpy_path = os.path.dirname(__file__)
codpy_path = os.path.join(codpy_path,"codpy")

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files(codpy_path)


DISTNAME = 'codpy'
DESCRIPTION = 'An RKHS based module for machine learning and data mining'
#with open('README.rst') as f:
  #  LONG_DESCRIPTION = f.read()
MAINTAINER = 'jean-marc mercier'
MAINTAINER_EMAIL = 'jeanmarc.mercier@gmail.com'
URL = 'https://github.com/johnlem/codpy_alpha'
#DOWNLOAD_URL = 'https://github.com/johnlem/codpy_alpha'
LICENSE = 'new BSD'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/johnlem/codpy_alpha/issues',
    'Documentation': 'https://',
    'Source Code': 'https://github.com/johnlem/codpy_alpha'
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
    packages=['codpy','codpy.Clustering'],
    include_package_data=True,
    package_data={'': extra_files},
    classifiers=[
        # trove classifiers
        # the full list is here: https://pypi.python.org/pypi?%3aaction=list_classifiers
        'development status :: alpha',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering'
        'Programming Language :: Python :: 3.7',
        'Operating System :: Microsoft :: Windows',
    ],
    install_requires=['pybind11','pandas>=1.0','numpy>=1.18','matplotlib>=3.2','mkl==2021.2.0','scikit-learn==1.0.2','scipy>=1.6.1',
    'tensorflow','seaborn', 'scikit-image','tensorflow-datasets','torch','xgboost','jupyter','quantlib','xlrd','pydicom','tk'],
    extras_require={
    'win32': 'pywin32'
  }
)
