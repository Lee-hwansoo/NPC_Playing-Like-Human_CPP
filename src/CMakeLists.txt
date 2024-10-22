add_subdirectory(npc)

find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(${PROJECT_NAME}_SOURCE_FILES
    main.cpp
)
add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCE_FILES}
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${${PROJECT_NAME}_INCLUDE_PATH}
    ${SDL2_INCLUDE_PATH}
    ${Eigen3_INCLUDE_PATH}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_NPC_LIBRARIES
    ${SDL2_LIBRARIES}
    ${Eigen3_LIBRARIES}
    ${CUDAToolkit_LIBRARIES}
    ${TORCH_LIBRARIES}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_options(${PROJECT_NAME} PRIVATE
    ${CPP_COMFILE_FLAGS}
)
if (MSVC)
  file(GLOB TORCH_DLLS "${Torch_INSTALL_PATH}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
