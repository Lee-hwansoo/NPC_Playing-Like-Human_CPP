set(${PROJECT_NAME}_NPC_FILES
    example.cpp
)
add_library(${PROJECT_NAME}_NPC STATIC
  ${${PROJECT_NAME}_NPC_FILES}
)
target_include_directories(${PROJECT_NAME}_NPC PUBLIC
  ${${PROJECT_NAME}_INCLUDE_PATH}
)
