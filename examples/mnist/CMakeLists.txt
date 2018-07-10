cmake_minimum_required(VERSION 3.0.0)

if (NEURAL_INFERENCE_ONLY)
    message(FATAL_ERROR "MNIST example requires AutoDiff using Stan Math for training")
endif()

# Define project
project(neural_mnist)

# Add MNIST dataset/parser repository
include(ExternalProject)
ExternalProject_Add(
        mnist_parser
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/mnist_parser
        GIT_REPOSITORY https://github.com/wichtounet/mnist
        GIT_TAG 6bb5922fbdeca4ce446e0d9182ea3a0697ea079e
        GIT_SHALLOW 1
        GIT_PROGRESS 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
)

# Add MNIST parser include files
ExternalProject_Get_Property(mnist_parser SOURCE_DIR)
set(MNIST_INCLUDE_DIR ${SOURCE_DIR}/include)
set(MNIST_DATA_DIR ${SOURCE_DIR})

# Create executable and link libraries
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_dependencies(${PROJECT_NAME} mnist_parser)
target_link_libraries(${PROJECT_NAME} PRIVATE neural)
target_include_directories(${PROJECT_NAME} PRIVATE ${MNIST_INCLUDE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
