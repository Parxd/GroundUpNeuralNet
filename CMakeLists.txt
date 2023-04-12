cmake_minimum_required(VERSION 3.0.0)
project(GroundUpNeuralNet VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

# Bring in GoogleTest
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929  # release-1.11.0
)
FetchContent_GetProperties(googletest)

if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

include_directories(${googletest_SOURCE_DIR}/googletest/include)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(
    SOURCE_FILES

    src/layers/Linear.cpp
    src/layers/activations/RELU.cpp
    src/layers/activations/Sigmoid.cpp
    src/layers/activations/Softmax.cpp
    src/losses/MSE.cpp

    tests/main.cpp
    tests/unit/LinearTest.cpp
    tests/unit/RELUTest.cpp

    )

add_executable(GroundUpNeuralNet ${SOURCE_FILES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
add_dependencies(GroundUpNeuralNet gtest)
target_link_libraries(GroundUpNeuralNet gtest ${CMAKE_THREAD_LIBS_INIT})
include(CPack)
