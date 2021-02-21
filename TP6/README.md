[![pipeline status](https://gitlab.inria.fr/bramas/cudacpu/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/cudacpu/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/cudacpu/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/cudacpu/commits/master)

CUDACPU is a simple CUDA emulator, created first for educational purposes, and second for debuging and developing CUDA kernel without NVidia GPUs.

The principle is simple, CUDACPU uses OpenMP to parallelize the code as it was a normal CPU code.
By doing so, a CUDA kernel becomes a simple/normal function for the regular CPU (and thus can be debugged as usual).

* It is important to notice that the project just starts, hence many features are missing *

# Compilation

Simply go in the build dir and use cmake as usual:
```
mkdir build
cd build
# To enable testing: cmake -DUSE_TESTING=ON ..
cmake ..
make
# To run the tests: make test
# To get the output of the tests: CTEST_OUTPUT_ON_FAILURE=TRUE make test
```

# Create your own kernel directly in CUDACPU

To do so simply create a new C++ file in the `tests` and rerun cmake to take it into account in the build process.

# Use CUDACPU in an existing project

You will need to include the `src` directory, which provides a `cuda.h` file.
Then consider that you were calling your cuda kernel with:
```cpp
my_kernel<<<X, Y>>>(p0, p1, p2);
```
It should be replace by:
```cpp
CudaCpu(X, Y, my_kernel, p0, p1, p2);
```

Note that when `cuda.h` is the one from CUDACPU, a macro `_CUDACPU_` is defined and can be use to facilitate the conditional building.
```cpp
#ifndef _CUDACPU_
my_kernel<<<X, Y>>>(p0, p1, p2);
#else
CudaCpu(X, Y, my_kernel, p0, p1, p2);
#endif
```

# Integration

## Gitlab ci

The file `.gitlab-ci.yml` contains the information related to the continuous integration on gitlab.

## Coverage result

Can be found here: https://bramas.gitlabpages.inria.fr/cudacpu/

