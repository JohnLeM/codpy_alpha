import os,sys,fnmatch,subprocess
from setuptools import setup, Extension, Command
from distutils.command.build_py import build_py as _build_py

__version__ = '0.0.1'

DISTNAME = 'codpy'
DESCRIPTION = 'An RKHS based module for machine learning and data mining'
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


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def get_wheel_path(pattern = "*.whl"):
    parent_path = os.path.dirname(__file__)
    wheel_path = [file for file in find_files(parent_path,pattern)]
    print(wheel_path)
    if len(wheel_path): wheel_path = wheel_path[0]
    return wheel_path

class build_py(_build_py):
    """Specialized Python source builder."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", get_wheel_path()])

long_description = open("README.md","r")


setup(
    name=DISTNAME,
    version=__version__,
    author=MAINTAINER,
    maintainer=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
#    cmdclass={'build_py': build_py},
    long_description=long_description,
    long_description_content_type='text/markdown'
)

    