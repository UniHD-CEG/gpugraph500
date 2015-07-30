# CMakeLists.txt for G4CU project                                                                                                                                                                          
project(test-cuda-thrust-gdb)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

cuda_compile(HELPER_O helper.cu)
cuda_compile(DRIVER_O driver.cu OPTIONS -G)

cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})
