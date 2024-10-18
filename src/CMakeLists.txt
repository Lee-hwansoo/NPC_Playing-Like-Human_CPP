set(${PROJECT_NAME}_LIBRARY_FILES
    example.cpp
)
add_library(${PROJECT_NAME}_LIBRARY STATIC
  ${${PROJECT_NAME}_LIBRARY_FILES}
)
target_include_directories(${PROJECT_NAME}_LIBRARY PUBLIC
  ${${PROJECT_NAME}_INCLUDE_PATH}
)

set(${PROJECT_NAME}_SOURCE_FILES
    main.cpp
)
add_executable(${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCE_FILES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PROJECT_NAME}_LIBRARY
)
target_compile_options(${PROJECT_NAME} PRIVATE
    ${CPP_COMFILE_FLAGS}
)
