# CUDA Mesh projector

Projects triangular meshes with (ASTRA) cone and cone vector geometries.

## License
The software is released under the GNU General Public License v3.0 - See the LICENSE file for more details.

# How to use
### Dependencies
`NVIDIA display driver` >= 495,
`cuda` 10.x or 11.x,
`NVIDIA OptiX` 7.4 (for ray tracing projectors),
`g++-8` if cuda 10.x is used, `g++-9` if cuda 11.x is used,
`python` 3.x, `numpy`, `cython`, `trimesh` (optional, for reading STL models), `mayavi` (optional, used in examples for 3D plotting)
### Linux build
#### Python package
This will build the `cad-astra` package without OptiX projectors and install it into your current conda environment:
```
git clone git@github.com:cad-astra/cad-astra.git
cd cad-astra
./build_python.sh install
```
If you wish to preserve your conda environment unchanged, then simply build the package without `install` option and make it accessible via PYTHONPATH environment variable:
```
git clone git@github.com:cad-astra/cad-astra.git
cd cad-astra
./build_python.sh
cd build/python/lib*
export PYTHONPATH=$(pwd)
```
To build python package with OptiX projector, you need to specify OptiX 7.x root directory via OPTIX_DIR environment variable,
and build with environment variable WITH_OPTIX=TRUE:
```
cd <OptiX 7 root>
export OPTIX_DIR=$(pwd)
cd <cad-astra directory>
WITH_OPTIX=TRUE ./build_python.sh install
```
To check other build options and environment variables that control the build, run
```
./build_python.sh --help
```
### Usage
Import `mesh_projector`:
```
import mesh_projector as mp
```
Create cuda arrays for vertices, normals, faces, projection angles (views) and sinogram from numpy arrays:
```
vert_id = mp.create_cuda_array('vertices', nd_array=vertices)
face_id = mp.create_cuda_array('faces', nd_array=faces)
norm_id = mp.create_cuda_array('normals', nd_array=normals)
ang_id = mp.create_cuda_array('angles', nd_array=angles)
det_rows, det_cols = 512, 512
sino_id = mp.create_cuda_array('sino', shape=(angles.shape[0], det_rows, det_cols))
```
Create projection geometry, e.g., 'cone' geometry:
```
det_size_x, det_size_y = 1, 1
proj_geom = mp.create_projection_geometry('cone', det_size_x, det_size_y, det_rows, det_cols, ang_id, 5000, 200)
```
Project:
```
mp.project(vert_id, face_id, norm_id, sino_id, proj_geom)

```
Copy sinogram to host memory and reshape it:
```
sino = mp.get_cuda_array(sino_id)
sino = sino.reshape((angles.shape[0], det_rows, det_cols))
```
See the `examples` directory for more examples.

### Running CUDA profiler
`nvprof` requires root priviliges for its work. To run it, create a bash script that starts `nvprof` and run it with sudo.
Example of `nvprof` invocation script:
```
#! /bin/bash -x

nvprof --track-memory-allocations on <script_that_runs_projector>.py
```
