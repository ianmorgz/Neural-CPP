#include "../inc/tensor.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace neuralcpp {
namespace math {

// basic math operations for tensors
void matmul(const Tensor& a, const Tensor& b, Tensor& out) { // matrix multiplication for 2D tensors
    if (a.ndim() != 2 || b.ndim() != 2) { // ensure we are working with 2D tensors
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    // get the dimensions of each tensor
    size_t a_num_rows = a.shape()[0];
    size_t a_num_cols = a.shape()[1];
    size_t b_num_rows = b.shape()[0];
    size_t b_num_cols = b.shape()[1];

    if(a_num_cols != b_num_rows){ // make sure our tensors are compatible for multiplication
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    if(out.shape() != std::vector<size_t>{a_num_rows, b_num_cols}){ // adjust output tensor shape if needed
        out = Tensor({a_num_rows, b_num_cols});
    }

    // perform matrix multiplication
    for(size_t i = 0; i < a_num_rows; ++i){
        for(size_t j = 0; j < b_num_cols; ++j){
            float sum = 0.0f;
            for(size_t k = 0; k < a_num_cols; ++k){
                sum += a.at({i, k}) * b.at({k, j});
            }
            out.at({i, j}) = sum;
        }
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }

    if (out.shape() != a.shape()) {
        out = Tensor(a.shape());
    }

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

void multiply(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication");
    }

    if (out.shape() != a.shape()) {
        out = Tensor(a.shape());
    }

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
}


// --- Activation Functions ---
void relu(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        x[i] = std::max(0.0f, x[i]);
    }
}

void sigmoid(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void tanh(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        x[i] = std::tanh(x[i]);
    }
}

void softmax(Tensor& x){
    throw std::runtime_error("Softmax derivative not implemented.");
}

void relu_derivative(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        x[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void sigmoid_derivative(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        float sig = 1.0f / (1.0f + std::exp(-x[i]));
        x[i] = sig * (1.0f - sig);
    }
}

void tanh_derivative(Tensor& x){
    for(size_t i = 0; i < x.size(); ++i){
        x[i] = 1.0f - std::tanh(x[i]) * std::tanh(x[i]);
    }
}

void softmax_derivative(Tensor& x){
    throw std::runtime_error("Softmax derivative not implemented.");
}

// --- loss functions ---
float mse_loss(const Tensor& predictions, const Tensor& targets){
    if(predictions.shape() != targets.shape()){
        throw std::invalid_argument("Predictions and targets must have the same shape for MSE loss calculation.");
    }

    float sum = 0.0f;
    for(size_t i = 0; i < predictions.size();i++){
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }

    return sum / static_cast<float>(predictions.size());
}

Tensor mse_loss_derivative(const Tensor& predictions, const Tensor& targets){
    if(predictions.shape() != targets.shape()){
        throw std::invalid_argument("Predictions and targets must have the same shape for MSE loss calculation.");
    }

    Tensor result(predictions.shape());

    for(size_t i = 0; i < predictions.size(); i++){
        result[i] = 2.0f * (predictions[i] - targets[i]) / static_cast<float>(predictions.size());
    }  

    return result;
}

void initialize_weights_zero(Tensor& tensor) {
    tensor.fill(0.0f);
}

void initialize_weights_random_uniform(Tensor& tensor, float lower, float upper) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(lower, upper);

    for(size_t i = 0; i < tensor.size(); i++){
        tensor[i] = distribution(generator);
    }
}

void initialize_weights_random_normal(Tensor& tensor, float mean, float stddev) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stddev);

    for(size_t i = 0; i < tensor.size(); i++){
        tensor[i] = distribution(generator);
    }
}

void initialize_weights_xavier(Tensor& tensor, size_t input_size, size_t output_size) {
    float limit = std::sqrt(6.0f / (input_size + output_size));
    initialize_weights_random_uniform(tensor, -limit, limit);
}

void initialize_weights_xavier_normal(Tensor& tensor, size_t input_size, size_t output_size) {
    float stddev = std::sqrt(2.0f / (input_size + output_size));
    initialize_weights_random_normal(tensor, 0.0f, stddev);
}

void initialize_weights_he_uniform(Tensor& tensor, size_t input_size) {
    float limit = std::sqrt(6.0f / input_size);
    initialize_weights_random_uniform(tensor, -limit, limit);
}

void initialize_weights_he_normal(Tensor& tensor, size_t input_size) {
    float stddev = std::sqrt(2.0f / input_size);
    initialize_weights_random_normal(tensor, 0.0f, stddev);
}

// bias initialization functions
void initialize_bias_zero(Tensor& tensor) {
    tensor.fill(0.0f);
}

void initialize_bias_constant(Tensor& tensor, float value) {
    tensor.fill(value);
}

void initialize_bias_uniform(Tensor& tensor, float lower, float upper) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(lower, upper);

    for(size_t i = 0; i < tensor.size(); i++){
        tensor[i] = distribution(generator);
    }
}

void initialize_bias_smart_output(Tensor& tensor) {
    // TODO initialize based on activation function
    // For output layers, initializing biases to a small negative value can help
    tensor.fill(-1.5f); // Example value; adjust based on activation function
}

} // namespace math
} // namespace neuralcpp