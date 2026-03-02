cmake_minimum_required(VERSION 3.18)
project(leviathan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# SM86 (3080 Ti) + SM90 (H100)
set(CMAKE_CUDA_ARCHITECTURES "86;90")

find_package(pybind11 REQUIRED)

# Shared CUDA flags — NO --use_fast_math (breaks reproducibility)
set(CUDA_OPTS
  $<$<COMPILE_LANGUAGE:CUDA>:
    --maxrregcount=64
    -lineinfo
    --expt-relaxed-constexpr
  >
)

# Target 1: Core physics engine
pybind11_add_module(leviathan_cuda
  src/cuda/leviathan_kernel.cu
  src/bindings/leviathan_pybind11.cpp
)
target_compile_options(leviathan_cuda PRIVATE ${CUDA_OPTS})

# Target 2: GPU Φ solver
pybind11_add_module(leviathan_phi
  src/cuda/leviathan_phi_kernel.cu
  src/bindings/leviathan_phi_pybind.cpp
)
target_compile_options(leviathan_phi PRIVATE ${CUDA_OPTS})

# Target 3: Multi-GPU (optional, requires NCCL)
option(ENABLE_MULTIGPU "Build multi-GPU NCCL engine" OFF)
if(ENABLE_MULTIGPU)
  find_package(NCCL REQUIRED)
  add_library(leviathan_multigpu SHARED
    src/cuda/leviathan_multigpu.cu
  )
  target_link_libraries(leviathan_multigpu PRIVATE nccl cuda)
  target_compile_options(leviathan_multigpu PRIVATE ${CUDA_OPTS})
endif()