add_subdirectory(npc)

find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)

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
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_NPC
    ${SDL2_LIBRARIES}
    ${Eigen3_LIBRARIES}
)
target_compile_options(${PROJECT_NAME} PRIVATE
    ${CPP_COMFILE_FLAGS}
)
