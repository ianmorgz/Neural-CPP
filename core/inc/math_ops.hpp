#pragma once
#include "tensor.hpp"

namespace neuralcpp {
namespace math {
// basic math operations for tensors
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
void add(const Tensor& a, const Tensor& b, Tensor& out);
void multiply(const Tensor& a, const Tensor& b, Tensor& out);

// activation functions
void relu(Tensor& x);
void sigmoid(Tensor& x);
void tanh(Tensor& x);
void softmax(Tensor& x);

void relu_derivative(Tensor& x);
void sigmoid_derivative(Tensor& x);
void tanh_derivative(Tensor& x);
void softmax_derivative(Tensor& x);

// loss functions
float mean_squared_error(const Tensor& predictions, const Tensor& targets);
Tensor mse_loss_derivative(const Tensor& predictions, const Tensor& targets);

// basic utilities
void initialize_he_normal(Tensor& tensor, size_t input_size);
void initialize_xavier(Tensor& tensor, size_t input_size, size_t output_size);

} // namespace math
} // namespace neuralcpp