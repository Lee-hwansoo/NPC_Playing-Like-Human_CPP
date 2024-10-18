cmake_minimum_required(VERSION 3.11)

project(
    NPC
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "Playing Like Human"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_CONFIGURATION_TYPES Debug Release)

message(STATUS "Started all process in ${PROJECT_NAME} CMakeLists.txt.\n")

set(CPP_COMPILE_FLAGS_MSVC /W4 /WX /permissive-)
set(CPP_COMPILE_FLAGS_OTHERS -Wall -Wpedantic -Wextra -Werror)

if(NOT CPP_COMPILE_FLAGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CPP_COMPILE_FLAGS ${CPP_COMPILE_FLAGS_MSVC})
    else()
        set(CPP_COMPILE_FLAGS ${CPP_COMPILE_FLAGS_OTHERS})
    endif()
endif()

set(${PROJECT_NAME}_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PROJECT_NAME}_SOURCE_PATH "${${PROJECT_NAME}_ROOT_PATH}/src")
set(${PROJECT_NAME}_INCLUDE_PATH "${${PROJECT_NAME}_ROOT_PATH}/include")
set(${PROJECT_NAME}_THIRDPARTY_PATH "${${PROJECT_NAME}_ROOT_PATH}/thirdparty")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(PRINT_SYSTEM_INFO "Print system information" ON)
option(PRINT_PROJECT_INFO "Print project information" ON)

if(PRINT_SYSTEM_INFO)
    include(cmake/print_system_info.cmake)
endif()
if(PRINT_PROJECT_INFO)
    include(cmake/print_project_info.cmake)
endif()

add_subdirectory(${${PROJECT_NAME}_THIRDPARTY_PATH})
add_subdirectory(${${PROJECT_NAME}_SOURCE_PATH})

message(STATUS "Finished all process in ${PROJECT_NAME} CMakeLists.txt.\n")
