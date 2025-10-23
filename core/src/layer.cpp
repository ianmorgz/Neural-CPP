#include "inc/layer.hpp"
#include "inc/math_ops.hpp"
#include <stdexcept>
#include <cmath>

namespace neuralcpp {

DenseLayer::DenseLayer(size_t input_size, size_t output_size, Activation activation )
: weights_( {input_size, output_size} ), biases_( {output_size} ), activation_(activation)
{
    // Initialize weights using He normal initialization for ReLU and Xavier for others
    if(activation == Activation::ReLU){
        math::initialize_he_normal(weights_, input_size);
    } else {
        math::initialize_xavier(weights_, input_size, output_size);
    }
    // Initialize biases to zero
    biases_.randomize(-0.1f, 0.1f);
}

Tensor DenseLayer::forward(const Tensor& input) {
    if(input.shape() != std::vector<size_t>{weights_.shape()[0]}){
        throw std::invalid_argument("Input shape does not match layer's expected input size.");
    }

    input_cache_ = input; // Cache input for backpropagation

    Tensor output({weights_.shape()[1]});

    for(size_t i = 0;i<output.shape()[0];i++){
        float sum = biases_[1];
        for(size_t j = 0;j < weights_.shape()[0];j++){
            sum += input[j] * weights_.at({j, i});
        }
        output[i] = sum;
    }

    output_cache_ = output;
    apply_activation(output); //TODO: implement apply_activation

    return output;
}

Tensor DenseLayer::backward(const Tensor& grad_output, float learning_rate) {
    // grad_output: derivative of loss with respect to this layer's output

    Tensor activation_grad = output_cache_; // Copy cached output
    apply_activation_derivative(activation_grad); //TODO: implement apply_activation_derivative

    Tensor grad_activation(grad_output.shape());
    for(size_t i = 0; i < activation_grad.size(); ++i){
        grad_activation[i] = grad_output[i] * activation_grad[i];
    }

    // Compute gradient for weights and biases
    Tensor weight_grad({weights_.shape()[0], weights_.shape()[1]});
    Tensor bias_grad({biases_.size()});
    
    // Compute gradients
    for (size_t i = 0; i < weights_.shape()[1]; ++i) {  // Output neurons
        bias_grad[i] = grad_activation[i];
        for (size_t j = 0; j < weights_.shape()[0]; ++j) {  // Input neurons
            weight_grad.at({j, i}) = input_cache_[j] * grad_activation[i];
        }
    }
    
    // Compute gradient for previous layer
    Tensor grad_input({input_cache_.size()});
    
    for (size_t i = 0; i < input_cache_.size(); ++i) {  // Input neurons
        float sum = 0.0f;
        for (size_t j = 0; j < weights_.shape()[1]; ++j) {  // Output neurons
            sum += weights_.at({i, j}) * grad_activation[j];
        }
        grad_input[i] = sum;
    }
    
    // Update weights and biases
    for (size_t i = 0; i < weights_.shape()[0]; ++i) {
        for (size_t j = 0; j < weights_.shape()[1]; ++j) {
            weights_.at({i, j}) -= learning_rate * weight_grad.at({i, j});
        }
    }
    
    for (size_t i = 0; i < biases_.size(); ++i) {
        biases_[i] -= learning_rate * bias_grad[i];
    }
    
    return grad_input;
}

void DenseLayer::set_weights(const Tensor& weights){
    if(weights.shape() != weights_.shape()){
        throw std::invalid_argument("Weights shape does not match layer's weights shape.");
    }
    weights_ = weights;
}

void DenseLayer::set_biases(const Tensor& biases){
    if(biases.shape() != biases_.shape()){
        throw std::invalid_argument("Biases shape does not match layer's biases shape.");
    }
    biases_ = biases;;
}

// float DenseLayer::activation_function(float x) const {
//     switch (activation_) {
//         case Activation::ReLU:
//             return std::max(0.0f, x);
//         case Activation::Sigmoid:
//             return 1.0f / (1.0f + std::exp(-x));
//         case Activation::Tanh:
//             return std::tanh(x);
//         case Activation::Softmax:
//             throw std::logic_error("Softmax should be applied to the entire tensor, not each individual element.");
//         case Activation::Linear:
//             return x;
//         default:
//             throw std::invalid_argument("Unknown activation function.");
//     }
// }

// float DenseLayer::activation_derivative(float x) const {
//     switch (activation_) {
//         case Activation::ReLU:
//             return x > 0.0f ? 1.0f : 0.0f;
//         case Activation::Sigmoid: {
//             float sig = activation_function(x);
//             return sig * (1 - sig);
//         }
//         case Activation::Tanh: {
//             float t = std::tanh(x);
//             return 1 - t * t;
//         }
//         case Activation::Softmax:
//             throw std::logic_error("Softmax derivative should be handled differently.");
//         case Activation::Linear:
//             return 1.0f;
//         default:
//             throw std::invalid_argument("Unknown activation function.");
//     }
// }

void DenseLayer::apply_activation(Tensor& tensor) const {
    switch (activation_) {
        case Activation::ReLU:
            return x > 0.0f ? 1.0f : 0.0f;
        case Activation::Sigmoid: {
            float sig = activation_function(x);
            return sig * (1 - sig);
        }
        case Activation::Tanh: {
            float t = std::tanh(x);
            return 1 - t * t;
        }
        case Activation::Softmax:
            throw std::logic_error("Softmax derivative should be handled differently.");
        case Activation::Linear:
            return 1.0f;
        default:
            throw std::invalid_argument("Unknown activation function.");
    }
}

void DenseLayer::apply_activation_derivative(Tensor& tensor) const {
    for(size_t i = 0; i < tensor.size(); i++){
        tensor[i] = activation_derivative(tensor[i]);
    }
}

} // namespace neuralcpp