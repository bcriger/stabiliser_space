from __future__ import absolute_import
from setuptools import setup, find_packages, Extension

import sys, os
from six.moves import map
sys.path.insert(0, os.path.join(os.getcwd(), 'src/'))

from stabiliser_space import __version__ as v

config = {
    'description': 'sparse_pauli-based implementation of common stabiliser operations',
    'author': 'Ben Criger',
    'url': 'https://github.com/bcriger/stabiliser_space',
    'download_url': 'https://github.com/bcriger/stabiliser_space.git',
    'author_email': 'bcriger@gmail.com',
    'version': '.'.join(map(str, v)),
    'install_requires': ['nose', 'sparse_pauli'],
    'package_data': {'': '*.so'},
    'package_dir': {'': 'src'},
    'packages': ['stabiliser_space'],
    'scripts': [],
    'name': 'stabiliser_space'
}

setup(**config)
