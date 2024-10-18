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
