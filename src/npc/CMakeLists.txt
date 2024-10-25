find_package(SDL2 REQUIRED HINTS ${SDL2_CMAKE_PATH})
find_package(Eigen3 REQUIRED HINTS ${Eigen3_CMAKE_PATH})

if(WIN32)
    find_package(CUDAToolkit REQUIRED)
    find_package(Torch REQUIRED HINTS ${Torch_CMAKE_PATH})
    enable_language(CUDA)
elseif(APPLE)
    find_package(Torch REQUIRED HINTS ${Torch_CMAKE_PATH})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(COMMON_INCLUDE_DIRS
    ${${PROJECT_NAME}_INCLUDE_PATH}
    ${SDL2_INCLUDE_PATH}
    ${Eigen3_INCLUDE_PATH}
    ${TORCH_INCLUDE_DIRS}
)

set(COMMON_LIBRARIES
    ${SDL2_LIBRARIES}
    ${Eigen3_LIBRARIES}
    ${TORCH_LIBRARIES}
)

if(WIN32)
    list(APPEND COMMON_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND COMMON_LIBRARIES ${CUDAToolkit_LIBRARIES})
endif()

# set(${PROJECT_NAME}_EXAMPLE_FILES
#     example.cpp
# )
# add_library(${PROJECT_NAME}_EXAMPLE_LIBRARIES STATIC
#   ${${PROJECT_NAME}_EXAMPLE_FILES}
# )
# target_include_directories(${PROJECT_NAME}_EXAMPLE_LIBRARIES PUBLIC
#   ${COMMON_INCLUDE_DIRS}
# )

set(${PROJECT_NAME}_MAIN_FILES
    test.cpp
    actor.cpp
    critic.cpp
    sac.cpp
    object.cpp
)
add_executable(${PROJECT_NAME}_MAIN
    ${${PROJECT_NAME}_MAIN_FILES}
)
target_include_directories(${PROJECT_NAME}_MAIN PRIVATE
    ${COMMON_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_MAIN PRIVATE
    ${COMMON_LIBRARIES}
)
set_property(TARGET ${PROJECT_NAME}_MAIN PROPERTY CXX_STANDARD 17)
target_compile_options(${PROJECT_NAME}_MAIN PRIVATE
    ${CPP_COMFILE_FLAGS}
)

if (MSVC)
  file(GLOB TORCH_DLLS "${Torch_INSTALL_PATH}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}_MAIN
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}_MAIN>)
endif (MSVC)
