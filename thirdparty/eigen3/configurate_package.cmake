project(
  Eigen3
  LANGUAGES CXX
  VERSION 3.4.0
  DESCRIPTION "C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms"
  HOMEPAGE_URL "https://gitlab.com/libeigen/eigen.git"
)

set(${PROJECT_NAME}_GIT_TAG 3.4.0 CACHE STRING "Eigen3 git tag")
set(${PROJECT_NAME}_PREFIX ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-prefix CACHE STRING "Eigen3 install prefix")
set(${PROJECT_NAME}_INSTALL_PATH ${CMAKE_SOURCE_DIR}/thirdparty/install/${CMAKE_BUILD_TYPE}/${PROJECT_NAME} CACHE STRING "Eigen3 install path")
set(${PROJECT_NAME}_CMAKE_PATH ${${PROJECT_NAME}_INSTALL_PATH}/lib/cmake/eigen3/ CACHE STRING "Eigen3 cmake path")
set(${PROJECT_NAME}_CMAKE_ARGS
-DCMAKE_INSTALL_PREFIX=${${PROJECT_NAME}_INSTALL_PATH}
-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
-DBUILD_SHARED_LIBS=OFF)

set(${PROJECT_NAME}_INCLUDE_PATH ${${PROJECT_NAME}_INSTALL_PATH}/include/eigen3 CACHE STRING "Eigen3 include path")
set(${PROJECT_NAME}_LIBRARIES Eigen3::Eigen CACHE STRING "Eigen3 library path")
