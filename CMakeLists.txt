cmake_minimum_required(VERSION 3.18)
project(leviathan LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Global Settings
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Link-Time Optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Target H100 (SM90) with backwards compat to A100 (SM80)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

# Pybind11 (required for Python bindings)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Shared CUDA flags
# ---------------------------------------------------------------------------
set(CUDA_OPTS
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --maxrregcount=64
    -lineinfo
    --expt-relaxed-constexpr
  >
)

# ---------------------------------------------------------------------------
# Target 1: Core Physics Engine (leviathan_cuda)
# ---------------------------------------------------------------------------

pybind11_add_module(leviathan_cuda
  src/cuda/leviathan_kernel.cu
  src/bindings/leviathan_pybind11.cpp
)
target_compile_options(leviathan_cuda PRIVATE ${CUDA_OPTS})

# ---------------------------------------------------------------------------
# Target 2: GPU Φ Solver (leviathan_phi)
# ---------------------------------------------------------------------------

pybind11_add_module(leviathan_phi
  src/cuda/leviathan_phi_kernel.cu
  src/bindings/leviathan_phi_pybind.cpp
)
target_compile_options(leviathan_phi PRIVATE ${CUDA_OPTS})

# ---------------------------------------------------------------------------
# Target 3: Multi-GPU Engine (optional, requires NCCL)
# ---------------------------------------------------------------------------

option(ENABLE_MULTIGPU "Build multi-GPU NCCL engine" OFF)
if(ENABLE_MULTIGPU)
  find_package(NCCL REQUIRED)
  add_library(leviathan_multigpu SHARED
    src/cuda/leviathan_multigpu.cu
  )
  target_link_libraries(leviathan_multigpu PRIVATE nccl cuda)
  target_compile_options(leviathan_multigpu PRIVATE ${CUDA_OPTS})
endif()