from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(["hippocampus/geometry_utils.pyx", "hippocampus/fastBVC.pyx"],
                          compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()], requires=['matplotlib', 'seaborn', 'scipy', 'tqdm']
)
