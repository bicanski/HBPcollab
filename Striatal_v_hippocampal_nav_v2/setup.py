from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize(["geometry_utils.pyx", "fastBVC.pyx"]),
    include_dirs=[numpy.get_include()]
)
