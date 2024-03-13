from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup( ext_modules = cythonize(Extension(
            "ER",
            sources=["ER.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],
            
     )))