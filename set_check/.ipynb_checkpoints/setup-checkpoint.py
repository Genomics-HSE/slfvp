from distutils.core import setup, Extension
from Cython.Build import cythonize

setup( ext_modules = cythonize(Extension(
            "pyset",
            sources=["pyset.pyx"],
            extra_compile_args=["-std=c++17"],
            language="c++"
     )))