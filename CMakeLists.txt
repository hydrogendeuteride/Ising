cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 87)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(Ising CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit)

add_executable(Ising main.cu)

set_target_properties(Ising PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
