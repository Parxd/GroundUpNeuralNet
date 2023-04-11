cmake_minimum_required(VERSION 3.0.0)
project(GroundUpNeuralNet VERSION 0.1.0)

include(CTest)
enable_testing()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(
    SOURCE_FILES
    tests/main.cpp
    src/Layer.cpp
    
    )

add_executable(GroundUpNeuralNet ${SOURCE_FILES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
