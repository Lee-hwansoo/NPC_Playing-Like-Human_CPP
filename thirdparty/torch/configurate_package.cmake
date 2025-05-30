project(
  Torch
  LANGUAGES CXX
  VERSION 2.5.0
  DESCRIPTION "libtorch"
  HOMEPAGE_URL "https://pytorch.org/"
)

set(${PROJECT_NAME}_INSTALL_PATH ${CMAKE_SOURCE_DIR}/thirdparty/install/${CMAKE_BUILD_TYPE}/${PROJECT_NAME} CACHE STRING "Torch install path")
set(${PROJECT_NAME}_CMAKE_PATH ${${PROJECT_NAME}_INSTALL_PATH}/share/cmake/Torch/ CACHE STRING "Torch cmake path")
set(${PROJECT_NAME}_CMAKE_ARGS
-DCMAKE_INSTALL_PREFIX=${${PROJECT_NAME}_INSTALL_PATH})

if(WIN32)
  # Windows + CUDA 12.4 + CUDA Toolkit 11.8
  set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" CACHE STRING "CUDA Toolkit path")
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe" CACHE STRING "CUDA compiler path")
  set(CUDA_NVCC_EXECUTABLE "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe" CACHE STRING "CUDA exe path")
  set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include" CACHE STRING "CUDA include path")
  set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib" CACHE STRING "CUDA cudart library path")
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

elseif(APPLE)
  # M1 Mac
  set(USE_CUDA OFF CACHE BOOL "Use CUDA")

endif()
