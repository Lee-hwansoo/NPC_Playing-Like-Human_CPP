find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

enable_language(CUDA)

set(${PROJECT_NAME}_NPC_FILES
    example.cpp
)
add_library(${PROJECT_NAME}_NPC STATIC
  ${${PROJECT_NAME}_NPC_FILES}
)
target_include_directories(${PROJECT_NAME}_NPC PUBLIC
  ${${PROJECT_NAME}_INCLUDE_PATH}
  ${SDL2_INCLUDE_PATH}
  ${Eigen3_INCLUDE_PATH}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
