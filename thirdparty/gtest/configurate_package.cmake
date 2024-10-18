project(
  GTest
  LANGUAGES CXX
  VERSION 1.12.1
  DESCRIPTION "Google Test"
  HOMEPAGE_URL "https://github.com/google/googletest.git"
)

set(${PROJECT_NAME}_GIT_TAG release-1.12.1 CACHE STRING "GTest git tag")
set(${PROJECT_NAME}_PREFIX ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-prefix CACHE STRING "GTest install prefix")
set(${PROJECT_NAME}_INSTALL_PATH ${CMAKE_SOURCE_DIR}/thirdparty/install/${CMAKE_BUILD_TYPE}/${PROJECT_NAME} CACHE STRING "GTest install path")
set(${PROJECT_NAME}_CMAKE_PATH ${${PROJECT_NAME}_INSTALL_PATH}/lib/cmake/GTest CACHE STRING "GTest cmake path")
set(${PROJECT_NAME}_CMAKE_ARGS
-DCMAKE_INSTALL_PREFIX=${${PROJECT_NAME}_INSTALL_PATH}
-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

set(${PROJECT_NAME}_INCLUDE_PATH ${${PROJECT_NAME}_INSTALL_PATH}/include CACHE STRING "GTest include path")
set(${PROJECT_NAME}_LIBRARIES GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main CACHE STRING "GTest library path")
