add_subdirectory(npc)

set(${PROJECT_NAME}_SOURCE_FILES
    main.cpp
)
add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCE_FILES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PROJECT_NAME}_NPC
)
target_compile_options(${PROJECT_NAME} PRIVATE
    ${CPP_COMFILE_FLAGS}
)
