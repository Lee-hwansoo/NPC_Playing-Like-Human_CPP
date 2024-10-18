cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(
    CPP_PROJECT_TEMPLATE
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "C++ Project Template"
)

message(STATUS "Started all process in ${PROJECT_NAME} CMakeLists.txt.\n")

set(CPP_COMFILE_FLAGS_MSVC /W4 /WX /permissive-)
set(CPP_COMFILE_FLAGS_OTHERS -Wall -Wpedantic -Wextra -Werror)

if(NOT CPP_COMFILE_FLAGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CPP_COMFILE_FLAGS ${CPP_COMFILE_FLAGS_MSVC})
    else()
        set(CPP_COMFILE_FLAGS ${CPP_COMFILE_FLAGS_OTHERS})
    endif()
endif()

set(${PROJECT_NAME}_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PROJECT_NAME}_SOURCE_PATH "${${PROJECT_NAME}_ROOT_PATH}/src")
set(${PROJECT_NAME}_INCLUDE_PATH "${${PROJECT_NAME}_ROOT_PATH}/include")
set(${PROJECT_NAME}_THIRDPARTY_PATH "${${PROJECT_NAME}_ROOT_PATH}/thirdparty")
set(${PROJECT_NAME}_MODULE_PATH "${${PROJECT_NAME}_ROOT_PATH}/module")
set(${PROJECT_NAME}_TEST_PATH "${${PROJECT_NAME}_ROOT_PATH}/test")
set(${PROJECT_NAME}_APPLICATION_PATH "${${PROJECT_NAME}_ROOT_PATH}/app")
set(${PROJECT_NAME}_RESOURCE_PATH "${${PROJECT_NAME}_ROOT_PATH}/resource")

if(NOT PARSED_SYSTEM_INFO)
    message(STATUS "---------------CMake information---------------")
    message(STATUS "CMake version: ${CMAKE_VERSION}\n")

    message(STATUS "---------------System information---------------")
    message(STATUS "System Architecture name: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "System OS name: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "System OS version: ${CMAKE_SYSTEM_VERSION}\n")

    message(STATUS "---------------Compiler information---------------")
    message(STATUS "C++ Compiler id: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "C++ Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "C++ Compiler options: ${CPP_COMFILE_FLAGS}\n")

    message(STATUS "---------------C++ Standard information---------------")
    message(STATUS "C++ Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")
    message(STATUS "C++ Standard required: ${CMAKE_CXX_STANDARD_REQUIRED}")
    message(STATUS "C++ Extensions: ${CMAKE_CXX_EXTENSIONS}\n")
endif()

set(PARSED_SYSTEM_INFO TRUE)

message(STATUS "---------------Project information---------------")
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "${PROJECT_NAME}_VERSION: ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME}_DESCRIPTION: ${PROJECT_DESCRIPTION}\n")

message(STATUS "---------------Project path information---------------")
message(STATUS "${PROJECT_NAME}_ROOT_PATH: ${${PROJECT_NAME}_ROOT_PATH}")
message(STATUS "${PROJECT_NAME}_SOURCE_PATH: ${${PROJECT_NAME}_SOURCE_PATH}")
message(STATUS "${PROJECT_NAME}_INCLUDE_PATH: ${${PROJECT_NAME}_INCLUDE_PATH}")
message(STATUS "${PROJECT_NAME}_THIRDPARTY_PATH: ${${PROJECT_NAME}_THIRDPARTY_PATH}")
message(STATUS "${PROJECT_NAME}_MODULE_PATH: ${${PROJECT_NAME}_MODULE_PATH}")
message(STATUS "${PROJECT_NAME}_TEST_PATH: ${${PROJECT_NAME}_TEST_PATH}")
message(STATUS "${PROJECT_NAME}_APPLICATION_PATH: ${${PROJECT_NAME}_APPLICATION_PATH}")
message(STATUS "${PROJECT_NAME}_RESOURCE_PATH: ${${PROJECT_NAME}_RESOURCE_PATH}\n")

# ! Add sub directory
add_subdirectory(${${PROJECT_NAME}_THIRDPARTY_PATH})
add_subdirectory(${${PROJECT_NAME}_MODULE_PATH})
add_subdirectory(${${PROJECT_NAME}_APPLICATION_PATH})

# ! Add source files here
set(${PROJECT_NAME}_SOURCE_FILES
    src/main.cpp
)

# ! If you want to make a library, use the following code
# add_library(${PROJECT_NAME} STATIC
#   ${${PROJECT_NAME}_SOURCE_FILES}
# )

# ! If you want to make an executable, use the following code
add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCE_FILES}
)

# ! If you want to add include directories and link libraries, use the following code
target_include_directories(${PROJECT_NAME} PUBLIC
  ${${PROJECT_NAME}_INCLUDE_PATH}
)

# target_link_libraries(${PROJECT_NAME} PRIVATE
# ! Add libraries here
# )

# add_dependencies(${PROJECT_NAME}
# ! Add dependencies here
# )

target_compile_options(${PROJECT_NAME} PRIVATE
    ${CPP_COMFILE_FLAGS}
)

include(cmake/create_documents.cmake)

add_subdirectory(${${PROJECT_NAME}_TEST_PATH})

message(STATUS "Finished all process in ${PROJECT_NAME} CMakeLists.txt.\n")
