#pragma once
#include "tensor.hpp"

namespace neuralcpp {
namespace math {
// basic math operations for tensors
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
void add(const Tensor& a, const Tensor& b, Tensor& out);
void multiply(const Tensor& a, const Tensor& b, Tensor& out);
void normalize(Tensor& x, float lower=0.0f, float upper=1.0f);

// activation functions
void relu(Tensor& x);
void sigmoid(Tensor& x);
void tanh(Tensor& x);
void softmax(Tensor& x);

// activation function derivatives
void relu_derivative(Tensor& x);
void sigmoid_derivative(Tensor& x);
void tanh_derivative(Tensor& x);
void softmax_derivative(Tensor& x);

// loss functions
float mse_loss(const Tensor& predictions, const Tensor& targets);
Tensor mse_loss_derivative(const Tensor& predictions, const Tensor& targets);

float cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
Tensor cross_entropy_loss_derivative(const Tensor& predictions, const Tensor& targets);

// weight initialization functions
void initialize_weights_zero(Tensor& tensor);
void initialize_weights_xavier(Tensor& tensor, size_t input_size, size_t output_size);
void initialize_weights_random_uniform(Tensor& tensor, float lower, float upper);
void initialize_weights_random_normal(Tensor& tensor, float mean, float stddev);
void initialize_weights_xavier_normal(Tensor& tensor, size_t input_size, size_t output_size);
void initialize_weights_he_uniform(Tensor& tensor, size_t input_size);
void initialize_weights_he_normal(Tensor& tensor, size_t input_size);

// bias initialization functions
void initialize_bias_zero(Tensor& tensor);
void initialize_bias_constant(Tensor& tensor, float value);
void initialize_bias_uniform(Tensor& tensor, float lower, float upper);
void initialize_bias_smart_output(Tensor& tensor);

} // namespace math
} // namespace neuralcpp