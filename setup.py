from setuptools import setup
from cmake_setuptools import *

setup(name='rats',
      description='',
      version='0.0.0.dev0',
      ext_modules=[CMakeExtension('rats')],
      cmdclass={'build_ext': CMakeBuildExt}
      )
