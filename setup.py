# from setuptools import setup
# # from cmake_setuptools import *

# setup(
#       name='cpprats',
#       description='',
#       version='24.6.7',
#       install_requires=[
#         'cmake_setuptools'
#       ],
#       ext_modules=[CMakeExtension('cpprats', 'cpprats')],
#       cmdclass={'build_ext': CMakeBuildExt}
# )


# from setuptools import setup, find_packages
# from cmake_setuptools import *

# setup(
#     name='rats',
#     version='24.6.7',
#     description='',

#     ext_modules=[
#         CMakeExtension(
#             '_rats', sourcedir='cpprats'
#         ),
#     ],
#     cmdclass={'build_ext': CMakeBuildExt},
#     packages=[
#         '_rats',
#     ],
#     package_dir={'_rats':'cpprats'},
#     #find_packages(), #include=['rats', 'rats.*']),
# #     package_data={
# #         'rats': ['*.so', '*.dll', '*.dylib'],  # Include shared libraries in the package
# #     },
#     include_package_data=True,
#     zip_safe=False
# )

import os
import subprocess
from setuptools import setup, Extension
import sys
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', **kwargs):
        sources = self.get_all_sources(sourcedir)
        super().__init__(name, sources=sources, **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)
        self.get_all_sources
    
    @classmethod
    def get_all_sources(cls, folder):
        """Find all C/C++ source files in the `folder` directory."""
        sources = []
        for root, dirs, files in os.walk(folder):
            for name in files:
                sources.append(os.path.join(root, name))
        return sources
    
    

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={self.get_python_executable()}'
        ]
        build_args = ['--config', 'Release']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def get_python_executable(self):
        return os.environ.get('PYTHON_EXECUTABLE', sys.executable)


setup(
    name='rats',
    ext_modules=[CMakeExtension('_rats', sourcedir='cpprats')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['rats'],
    zip_safe=False,
)
