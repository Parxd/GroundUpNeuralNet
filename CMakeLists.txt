cmake_minimum_required(VERSION 3.24)
project(GNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories("./include")

add_library(Core STATIC)
add_subdirectory(./src)

# Build GTest automated testing suite
add_executable(test)
add_subdirectory(./tests)
add_dependencies(test gtest Core)
target_link_libraries(test gtest ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test Core)

add_executable(GNN ./src/main.cpp)
target_link_libraries(GNN Core)