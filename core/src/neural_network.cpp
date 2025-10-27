#include "../inc/neural_network.hpp"
#include "../inc/math_ops.hpp"
#include <fstream>
#include <stdexcept>

namespace neuralcpp{

NeuralNetwork::NeuralNetwork() : layers_(), last_output_() {} //default constructor

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer){
    layers_.push_back(std::move(layer));
}

Tensor NeuralNetwork::forward(const Tensor& input){
    Tensor current_input = input;
    for(const auto& layer : layers_){
        current_input = layer->forward(current_input);
    }
    last_output_ = current_input; // Cache the output for potential use
    return current_input;
}

void NeuralNetwork::backward(const Tensor& target, float learning_rate){
    if(last_output_.size() != target.size()){
        throw std::invalid_argument("Target size does not match the output size of the network.");
    }

    //find the inital gradient from loss derivative
    Tensor grad_output = math::mse_loss_derivative(last_output_, target);

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
    // TODO: Save number of layers


    // TODO: add full serialization of each layer's weights and biases
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

// Factory functions - exported names must match declarations in the header
std::unique_ptr<NeuralNetwork> createNeuralNetwork() {
    return std::make_unique<NeuralNetwork>();
}

std::unique_ptr<DenseLayer> createDenseLayer(size_t input_size, 
                                              size_t output_size, 
                                              Activation activation,
                                              WeightInitialization weight_init,
                                              BiasInitialization bias_init
                                            ) {
    return std::make_unique<DenseLayer>(input_size, output_size, activation, weight_init, bias_init);
}

} // namespace neuralcpp
