#include "inc/neural_network.hpp"
#include "inc/math_ops.hpp"
#include <fstream>
#include <stdexcept>

namespace neuralcpp{
NeuralNetwork::NeuralNetwork() : layers_(), last_input_() {} //default constructor
void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer){
    layers_.push_back(std::move(layer));
}
Tensor NeuralNetwork::forward(const Tensor& input){
    Tensor current_input = input;
    for(const auto& layer : layers_){
        current_input = layer->forward(current_input);
    }
    last_input_ = input; // Cache the input for potential use
    return current_input;
}

void NeuralNetwork::backward(const Tensor& target, float learning_rate){
    if(last_input_.size() != target.size()){
        throw std::invalid_argument("Target size does not match the output size of the network.");
    }

    //find the inital gradient from loss derivative
    Tensor grad_output = math::mse_loss_derivative(last_input_, target);

    //backpropogate through recursion
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad_output = layers_[i]->backward(grad_output, learning_rate);
    }
}

void NeuralNetwork::saveModel(const std::string& filepath) const{
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    size_t num_layers = layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    // Save number of layers
    size_t num_layers = layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // TODO add full serialization of each layer's weights and biases
    file.close();
}

void NeuralNetwork::loadModel(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    size_t num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    layers_.clear();
    // TODO add full deserialization of each layer's weights and biases
    file.close();
}

// Factory functions
std::unique_ptr<NeuralNetwork> create_network() {
    return std::make_unique<NeuralNetwork>();
}

std::unique_ptr<DenseLayer> create_dense_layer(size_t input_size, 
                                              size_t output_size, 
                                              Activation activation) {
    return std::make_unique<DenseLayer>(input_size, output_size, activation);
}

} // namespace neuralcpp
