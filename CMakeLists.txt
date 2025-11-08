cmake_minimum_required(VERSION 3.20)
project(lightweight_nn)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(lightweight_nn 
    core/src/tensor.cpp
    core/src/layer.cpp
    core/src/neural_network.cpp
    core/src/math_ops.cpp
    core/src/dataset.cpp
)

# Include directories
target_include_directories(lightweight_nn PUBLIC include)

# Examples
add_executable(xor_example core/examples/xor_example.cpp)
add_executable(parity_example core/examples/parity_example.cpp)
add_executable(mnist_handwriting_example core/examples/mnist_handwriting_example.cpp)
add_executable(loan_prediction_example core/examples/loan_prediction_example.cpp)
target_link_libraries(loan_prediction_example lightweight_nn)
target_link_libraries(mnist_handwriting_example lightweight_nn)
target_link_libraries(xor_example lightweight_nn)
target_link_libraries(parity_example lightweight_nn)