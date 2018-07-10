cmake_minimum_required(VERSION 3.0.0)

# Define project
project(neural)

# Store executables in bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Define options
set(STAN_MATH_PATH "/home/mathias/software/math" CACHE STRING "Path to the Stan Math repository")
option(NEURAL_INFERENCE_ONLY "Whether to turn off AutoDiff (training), this removes Stan Math as a dependency" OFF)
option(NEURAL_BUILD_EXAMPLES "Whether to build the neural examples" ON)

# Create neural library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)

if (NEURAL_INFERENCE_ONLY)
    # Inference only using Eigen
    find_package(Eigen3 REQUIRED)
    target_include_directories(${PROJECT_NAME} INTERFACE ${EIGEN3_INCLUDE_DIR})
else()
    # AutoDiff (training) using Stan Math and dependencies
    add_library(STAN_MATH INTERFACE)
    target_include_directories(STAN_MATH
            INTERFACE
            ${STAN_MATH_PATH}
            ${STAN_MATH_PATH}/lib/cvodes_2.9.0/include
            ${STAN_MATH_PATH}/lib/eigen_3.3.3
            ${STAN_MATH_PATH}/lib/boost_1.64.0
            )

    # Flag used to silence warnings in Boost 1.64
    target_compile_options(STAN_MATH INTERFACE -Wno-unused-local-typedefs -Wunused-function)

    # Use Stan Math for auto diff in neural
    target_link_libraries(${PROJECT_NAME} INTERFACE STAN_MATH)
    target_compile_definitions(${PROJECT_NAME} INTERFACE -DAUTO_DIFF_ENABLED)
endif()

if (NEURAL_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()

# Set C++11 and warning flags
target_compile_options(${PROJECT_NAME} INTERFACE -Wall)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

# Prepare Catch library for test executable
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

# Make test executable
set(TEST_NAME ${PROJECT_NAME}_tests)
add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralTests.cpp)
target_include_directories(${TEST_NAME} PRIVATE ${CATCH_INCLUDE_DIR})
target_link_libraries(${TEST_NAME} ${PROJECT_NAME} Catch)
