# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "rSGM",
        ["rSGM.pyx"],
        include_dirs       = [np.get_include()],
        extra_compile_args = ['-march=native', '-fopenmp', '-msse','-msse2','-mavx','-mavx2','-mssse3','-mfma','-mpopcnt'],
        extra_link_args    = ['-march=native' ,'-fopenmp','-g'],)
]

"""
ext_modules = [
    Extension(
        "C",
        ["C.pyx"],
    )
]
"""

setup(
    name='stereoMatch',
    ext_modules=cythonize(ext_modules, annotate=True),
)