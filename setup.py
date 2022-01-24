import os,sys
from setuptools import setup

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
wheel_path = os.path.dirname(__file__)
wheel_path = os.path.join(wheel_path,"dist",r"codpy-0.0.1-cp39-cp39-win_amd64.whl")


setup(
    name=DISTNAME,
    version=__version__,
    author=MAINTAINER,
    maintainer=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    classifiers=[
        # trove classifiers
        # the full list is here: https://pypi.python.org/pypi?%3aaction=list_classifiers
        'development status :: alpha',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering'
        'Programming Language :: Python :: 3.9.7',
        'Operating System :: Microsoft :: Windows',
    ],
    packages=[wheel_path]
)