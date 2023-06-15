#
# This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
# Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.
#
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import sys
import numpy
import os
import platform
import shutil

def directory_find(atom, root='.'):
    """
    Find the subdirectory `atom` in the `root` directory.
    Returns path to `atom` if found.
    Returns None if nothing was found.
    """
    for path, dirs, files in os.walk(root):
        if atom in dirs:
            return os.path.join(path, atom)
    return None

project_dir = r'./'
if os.environ.get('PROJ_DIR', ''):
    project_dir = os.environ['PROJ_DIR']
    print(f'Took project directory from os environment variable PROJ_DIR: {project_dir}')
pyx_dir     = project_dir + r'/python/mesh_projector/'
src_dir     = project_dir + r'/src/'

extra_opts = []
if platform.system() == 'Linux':
    if os.path.isfile('/usr/lib/x86_64-linux-gnu/libcudart.so'):
        cuda_lib_dir     = '/usr/lib/x86_64-linux-gnu/'
        cuda_include_dir ='/usr/include'
    elif os.path.isfile('/usr/local/cuda/lib/x64/libcudart.so'):
        cuda_lib_dir     = '/usr/local/cuda/lib/x64'
        cuda_include_dir = '/usr/local/cuda/include'
    elif os.path.isfile('/usr/local/cuda/lib64/libcudart.so'):
        cuda_lib_dir     = '/usr/local/cuda/lib64'
        cuda_include_dir = '/usr/local/cuda/include'
    else:
        print('Error: could not find CUDA folder. Is CUDA installed?')
        sys.exit(0)
elif platform.system() == 'Windows':
    extra_opts += ['/MT','/O2']
    if 'CUDA_PATH' in os.environ:
            cuda_dir = os.environ['CUDA_PATH']
            cuda_lib_dir     = cuda_dir + '/lib/x64'
            cuda_include_dir = cuda_dir + '/include'
    else:
            print('Error: environment variable CUDA_PATH is not set. Is CUDA installed?')
            sys.exit(0)

print(f'Found CUDA libs in {cuda_lib_dir} and CUDA headers in {cuda_include_dir}')

# mesh_projector_lib_dir = None
# if 'install' in sys.argv:
#     # mesh_projector_lib_dir=directory_find('mesh_projector', sys.exec_prefix)
#     if mesh_projector_lib_dir is None:
#         mesh_projector_lib_dir=sys.exec_prefix # TODO: this path is not where .so file will end up, it has to be fixed
# else:
#     mesh_projector_lib_dir=project_dir + '/build/lib'

# print(f".so files directory guess: {mesh_projector_lib_dir}")

include_dirs=[project_dir + '/include/', project_dir + 'src', cuda_include_dir, numpy.get_include()]
projector_define_macros = []
if os.environ.get('WITH_OPTIX', '') == 'TRUE':
    projector_define_macros.append(("WITH_OPTIX_PROJECTOR", None))
    optix_include_dir = os.environ.get('OPTIX_DIR', '') + '/include'
    include_dirs.append(optix_include_dir)

extensions = [Extension('mesh_projector.cuda_projector', sources=[pyx_dir + 'cuda_projector.pyx'],
                        libraries=['project_mesh', 'cudart'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        define_macros=projector_define_macros,
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        ),
              Extension('mesh_projector.cuda_managed_array', sources=[pyx_dir + 'cuda_managed_array.pyx'],
                        libraries=['project_mesh', 'cudart'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        ),
              Extension('mesh_projector.cuda_gpu_utils', sources=[pyx_dir + 'cuda_gpu_utils.pyx'],
                        libraries=['project_mesh', 'cudart'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        ),
              Extension('mesh_projector.global_config', sources=[pyx_dir + 'global_config.pyx'],
                        libraries=['project_mesh'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        ),
              Extension('mesh_projector.host_logging', sources=[pyx_dir + 'host_logging.pyx'],
                        libraries=['project_mesh'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        ),
              Extension('mesh_projector.cuda_interoperability', sources=[pyx_dir + 'cuda_interoperability.pyx'],
                        libraries=['project_mesh', 'cudart'], library_dirs = [cuda_lib_dir, project_dir + '/build/lib'],
                        language='c++', # extra_compile_args = ["-g", "-O0"],   # Uncomment for debug build
                        include_dirs=include_dirs,
                        # runtime_library_dirs=[mesh_projector_lib_dir]
                        )]

# ext_modules = cythonize(os.path.join(pyx_dir, '*.pyx'),
                        # language_level=3, force=True, language='c++')

# for m in ext_modules:
#     m.sources.append(os.path.join(src_dir, "object_manager.cpp"))

# DIRTY HACK that lets the cythonized extensions find .so files built by nvcc before setup.py
for path, dirs, files in os.walk(project_dir + '/build/lib/'):
    for f in files:
        shutil.copy(project_dir + '/build/lib/' + f, sys.exec_prefix + '/lib')

setup(cmdclass = {'build_ext': build_ext},
      name='mesh-fp-prototype',
      version='1.1.0dev',  # TODO (pavel): increment version
      author='Pavel Paramonov',
      author_email="pavel.paramonov@uantwerpen.be",
      packages=['mesh_projector'],
      package_dir={'mesh_projector': 'mesh_projector'},
      package_data={'mesh_projector': ['*.ptx']},
    #   data_files=[('lib', ['../build/lib/libproject_mesh.so', "../build/lib/libgpu_utils.so", "../build/lib/libglobal_config.so", "../build/lib/libobject_manager.so"])],
      ext_modules=cythonize(extensions, force=True)
)