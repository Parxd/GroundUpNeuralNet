cmake_minimum_required(VERSION 3.13)
project(GroundUpNeuralNet VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_FLAGS_TEST "${CMAKE_CXX_FLAGS_TEST} -DTEST_MODE")
set(CMAKE_CXX_FLAGS_TRAIN "${CMAKE_CXX_FLAGS_TRAIN} -DTRAIN_MODE")

add_executable(GNN_test ./tests/main.cpp) # test configuration
add_executable(GNN_train ./src/main.cpp) # train configuration

target_sources(GNN_test PRIVATE
        # Source files
        ./src/layers/Linear.cpp
        ./src/layers/RELU.cpp
        ./src/losses/CE.cpp
        ./src/losses/MSE.cpp

        ./src/containers/Container.cpp

        # Test files
        ./tests/unit/LinearTest.cpp
        ./tests/unit/ContainerTest.cpp
        )
target_sources(GNN_train PRIVATE
        # Source files
        ./src/layers/Linear.cpp
        ./src/layers/RELU.cpp
        ./src/losses/CE.cpp
        ./src/losses/MSE.cpp

        ./src/containers/Container.cpp
        )

target_compile_options(GNN_test PRIVATE -DTEST_MODE)
target_compile_options(GNN_train PRIVATE -DTRAIN_MODE)

# GTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG ccdeec888ebb740a7ea4e07d3e84a1b7ee32b315
)
FetchContent_GetProperties(googletest)
FetchContent_MakeAvailable(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()
MESSAGE(${googletest_SOURCE_DIR})
include_directories(${googletest_SOURCE_DIR}/googletest/include)

# Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen)
FetchContent_MakeAvailable(eigen)
if (NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR})
endif()
include_directories(${eigen_SOURCE_DIR}/eigen/include)

# Link Gtest + Eigen
target_link_libraries(GNN_test gtest_main eigen)
target_link_libraries(GNN_train eigen)