from setuptools import setup
from cmake_setuptools import *

setup(name='rats',
      description='',
      version='0.0.0.dev0',
      ext_modules=[CMakeExtension('rats')],
      cmdclass={'build_ext': CMakeBuildExt}
      )

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# from pathlib import Path
# from pybind11 import get_include as get_pybind_include

# ROOT_DIR = Path().absolute().parent

# print(get_pybind_include(), get_pybind_include(user=True))
# exit()

# rat_ext = Extension("rats",
#         ["rats_lib.pyx"],
#         language='c++',
#         include_dirs=[str(ROOT_DIR / 'include')],
#         extra_objects=[str(ROOT_DIR / 'build'/'librat.a')],
#         libraries=['spdlog'],
#         extra_compile_args=['-std=c++17', '-v'],
#     )
# rat_ext.cython_c_in_temp = True

# setup(
#     name='rats',
#     ext_modules=[rat_ext],
#     cmdclass = {'build_ext': build_ext},
# )
