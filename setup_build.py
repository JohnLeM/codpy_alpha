import os, sys

from distutils.core import setup
from distutils import sysconfig
from setuptools import setup, Extension, Command
from setuptools import setup, find_packages
from shutil import copy

__version__ = '0.0.1'

#define_macros = []
#extra_compile_args = []
#extra_link_args=[]
define_macros = []
extra_compile_args = ['/MT','/GL','/W3','/EHsc','/std:c++14','/bigobj']
extra_link_args=['/MANIFEST','/MACHINE:X64','/NXCOMPAT','/OPT:NOICF','/VERBOSE:LIB']
runtime_library_dirs=[os.getenv('PATH')]

#libraries = ['OTS_LibReleasex64','LinearAlgebraReleasex64']
#library_dirs=[os.path.join(os.getenv('BOOST'), 'build\\64'),os.path.join(os.getenv('QL'), 'lib'),os.path.join(os.getenv('SRC'), os.path.join(os.getenv('SRC'), 'lib','x64','Release'))]
#libraries = ['OTS_LibRelease (static runtime)x64','LinearAlgebraRelease (static runtime)x64','mkl_core','mkl_intel_thread','mkl_intel_lp64','libiomp5md']
libraries = ['OTS_LibRelease (static runtime)x64','LinearAlgebraRelease (static runtime)x64','mkl_rt','libiomp5md']
library_dirs=[os.getenv('MKL_LIB'),os.path.join(os.getenv('BOOST'), 'build\\64'),os.path.join(os.getenv('QL'), 'lib'),os.path.join(os.getenv('SRC'), os.path.join(os.getenv('SRC'), 'lib','x64','Release (static runtime)'))]
include_dirs=[os.getenv('QL'),os.getenv('MKL'),os.getenv('BOOST'),os.getenv('SRC'),os.getenv('PYBIND11'),os.getenv('XTENSOR')+'\include',os.getenv('NUMPY')]
print('############################')
print('############# define_macros')
print(define_macros)
print('############# include_dirs')
print(include_dirs)
print('############# library_dirs')
print(library_dirs)
print('############# libraries')
print(libraries)
print('############# extra_link_args')
print(extra_link_args)
print('############# extra_compile_args')
print(extra_compile_args)
print('############################')
codpy = Extension(
    name='codpy',
    sources=['src/codpy.cpp'],
    define_macros=define_macros,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args = extra_link_args
    )

setup(
    name='codpy',
    version=__version__,
    author='jean-marc mercier',
    author_email='jeanmarc.mercier@gmail.com',
    description='support vector machine utilities',
    url='https://github.com/johnlem/codpy/',
    ext_modules=[codpy],
    include_package_data=True,
    classifiers=[
        # trove classifiers
        # the full list is here: https://pypi.python.org/pypi?%3aaction=list_classifiers
        'development status :: 2 - pre-alpha',
    ]
)

#dest_file = sys.exec_prefix
#dest_file = os.path.join(dest_file, "Lib","site-packages","codpy.pyd")

#cp_file = os.path.dirname(os.path.realpath(__file__))
#cp_file = os.path.join(cp_file,"codpy","dist","codpy.pyd")

#print("copy(cp_file, dest_file), cp_file:",cp_file,", dest_file:",dest_file)

#copy(cp_file, dest_file)
