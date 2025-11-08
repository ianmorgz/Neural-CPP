#include "../inc/layer.hpp"
#include "../inc/math_ops.hpp"
#include <stdexcept>
#include <cmath>

namespace neuralcpp {

DenseLayer::DenseLayer(size_t input_size, size_t output_size, Activation activation, WeightInitialization weight_init, BiasInitialization bias_init)
: weights_( {input_size, output_size} ), biases_( {output_size} ), activation_(activation)
{
    // initialize_weights
    initialize_weights(weight_init);

    // initialize bias
    initialize_bias(bias_init);
}

Tensor DenseLayer::forward(const Tensor& input) {
    if(input.shape() != std::vector<size_t>{weights_.shape()[0]}){
        throw std::invalid_argument("Input shape does not match layer's expected input size.");
    }

    input_cache_ = input; // Cache input for backpropagation

    Tensor output({weights_.shape()[1]});

    for(size_t i = 0;i<output.shape()[0];i++){
        float sum = biases_[i]; // start with bias
        for(size_t j = 0;j < weights_.shape()[0];j++){
            sum += input[j] * weights_.at({j, i});
        }
        output[i] = sum;
    }

    output_cache_ = output;
    apply_activation(output); 

    return output;
}

Tensor DenseLayer::backward(const Tensor& grad_output, float learning_rate) {
    // grad_output: derivative of loss with respect to this layer's output

    Tensor activation_grad = output_cache_; // Copy cached output
    apply_activation_derivative(activation_grad); //TODO: implement apply_activation_derivative

    Tensor grad_activation(grad_output.shape()); //. multiply by activation derivative (chain rule)
    for(size_t i = 0; i < activation_grad.size(); ++i){
        grad_activation[i] = grad_output[i] * activation_grad[i];
    }

    // Compute gradient for weights and biases
    Tensor weight_grad({weights_.shape()[0], weights_.shape()[1]});
    Tensor bias_grad({biases_.size()});
    
    // Compute gradients
    for (size_t i = 0; i < weights_.shape()[0]; ++i) {  
        for (size_t j = 0; j < weights_.shape()[1]; ++j) {
            weight_grad.at({i, j}) = input_cache_[i] * grad_activation[j];
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
    biases_ = biases;
}

void DenseLayer::apply_activation(Tensor& tensor) const {
    switch (activation_) {
        case Activation::ReLU:
            math::relu(tensor);
            break;
        case Activation::Sigmoid: 
            math::sigmoid(tensor);
            break;
        case Activation::Tanh: 
            math::tanh(tensor);
            break;
        case Activation::Softmax: 
            math::softmax(tensor);
            break;
        default:
            throw std::invalid_argument("Unknown activation function.");
    }
}

void DenseLayer::apply_activation_derivative(Tensor& tensor) const {
    switch (activation_) {
        case Activation::ReLU:
            math::relu_derivative(tensor);
            break;
        case Activation::Sigmoid: 
            math::sigmoid_derivative(tensor);
            break;
        case Activation::Tanh: 
            math::tanh_derivative(tensor);
            break;
        case Activation::Softmax: 
            math::softmax_derivative(tensor);
            break;
        default:
            throw std::invalid_argument("Unknown activation function.");
    }
} 

void DenseLayer::initialize_weights(WeightInitialization weight_init){
    switch (weight_init){
        case WeightInitialization::Zero:
            math::initialize_weights_zero(weights_);
            break;
        case WeightInitialization::Xavier:
            math::initialize_weights_xavier(weights_, weights_.shape()[0], weights_.shape()[1]);
            break;
        case WeightInitialization::XavierNormal:
            math::initialize_weights_xavier_normal(weights_, weights_.shape()[0], weights_.shape()[1]);
            break;
        case WeightInitialization::HeUniform:
            math::initialize_weights_he_uniform(weights_, weights_.shape()[0]);
            break;
        case WeightInitialization::HeNormal:
            math::initialize_weights_he_normal(weights_, weights_.shape()[0]);
            break;
    }
}

void DenseLayer::initialize_bias(BiasInitialization bias_init){
    switch (bias_init){
        case BiasInitialization::Zero:
            math::initialize_bias_zero(biases_);
            break;
        case BiasInitialization::Constant:
            math::initialize_bias_constant(biases_, 0.0f);
            break;
        case BiasInitialization::Uniform:
            math::initialize_bias_uniform(biases_, -0.1f, 0.1f);
            break;
        case BiasInitialization::SmartOutput:
            math::initialize_bias_smart_output(biases_);
            break;
    }
}

} // namespace neuralcpp