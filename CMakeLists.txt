cmake_minimum_required(VERSION 2.8.11)

# Define project
project(neural)

# Define options
set(STAN_MATH_PATH "/home/mathias/software/math" CACHE STRING "Path to the Stan Math repository")

# Create neural library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)

# Stan math and dependencies
target_include_directories(${PROJECT_NAME} INTERFACE ${STAN_MATH_PATH})
target_include_directories(${PROJECT_NAME} INTERFACE ${STAN_MATH_PATH}/lib/cvodes_2.9.0/include)
target_include_directories(${PROJECT_NAME} INTERFACE ${STAN_MATH_PATH}/lib/eigen_3.3.3)
target_include_directories(${PROJECT_NAME} INTERFACE ${STAN_MATH_PATH}/lib/boost_1.64.0)

# Set C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-unused-local-typedefs")
add_definitions("-Wall")

# Prepare Catch library for test executable
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

# Make test executable
set(TEST_NAME ${PROJECT_NAME}_tests)
add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralTests.cpp)
target_include_directories(${TEST_NAME} PUBLIC ${CATCH_INCLUDE_DIR})
target_link_libraries(${TEST_NAME} ${PROJECT_NAME} Catch)
