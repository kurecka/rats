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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)

    def get_python_executable(self):
        return os.environ.get('PYTHON_EXECUTABLE', sys.executable)


setup(
    name='rats',
    ext_modules=[CMakeExtension('_rats', sourcedir='cpprats')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['rats'],
    zip_safe=False,
)
