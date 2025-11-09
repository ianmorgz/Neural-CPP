/*
--------------------------------------------------------------------------
neural_network.cpp - Implementation of the NeuralNetwork class
Where the core methods for forward and backward propagation are defined.
--------------------------------------------------------------------------
*/

#include "../inc/neural_network.hpp"
#include "../inc/math_ops.hpp"
#include <fstream>
#include <stdexcept>

namespace neuralcpp {

NeuralNetwork::NeuralNetwork() : layers_(), last_output_() {}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer){
    layers_.push_back(std::move(layer));
}

Tensor NeuralNetwork::forward(const Tensor& input){
    Tensor current_input = input;
    for(const auto& layer : layers_){
        current_input = layer->forward(current_input);
    }
    last_output_ = current_input;
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
    try{
        //save layer architecture info first
        // first save number of layers
        size_t num_layers = layers_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));

        // now iterate through each layer
        for(int i = 0; i < layers_.size(); ++i){
            Layer& layer = *layers_[i];
            const Tensor& weights = layer.get_weights();
            const Tensor& biases = layer.get_biases();
            int activation_type = layer.get_activation_type(); // assuming Layer has this method

            // save activation type
            file.write(reinterpret_cast<const char*>(&activation_type), sizeof(int));

            // save the input and output sizes
            size_t input_size = weights.shape()[0];
            size_t output_size = weights.shape()[1];
            file.write(reinterpret_cast<const char*>(&input_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&output_size), sizeof(size_t));

            // save weights data
            for(size_t j = 0; j < weights.size(); ++j){
                float weight_value = weights[j];
                file.write(reinterpret_cast<const char*>(&weight_value), sizeof(float));
            }

            // save biases data
            for(size_t j = 0; j < biases.size(); ++j){
                float bias_value = biases[j];
                file.write(reinterpret_cast<const char*>(&bias_value), sizeof(float));
            }
        }
        
        file.close();
    }catch (const std::exception& e){
        file.close();
        throw;
    }
}

void NeuralNetwork::loadModel(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }


    size_t num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    layers_.clear();
    for(int i = 0; i < num_layers; i++){
        // read activation type
        int activation_type_int = 0;
        file.read(reinterpret_cast<char*>(&activation_type_int), sizeof(int));
        Activation activation_type = static_cast<Activation>(activation_type_int);

        // read input and output sizes
        size_t input_size = 0;
        size_t output_size = 0;
        file.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));

        // create a new DenseLayer with default initializations
        auto layer = std::make_unique<DenseLayer>(input_size, output_size, activation_type, WeightInitialization::Xavier, BiasInitialization::Zero);

        // read weights data
        Tensor weights({input_size, output_size});
        for(size_t j = 0; j < weights.size(); ++j){
            float weight_value = 0.0f;
            file.read(reinterpret_cast<char*>(&weight_value), sizeof(float));
            weights[j] = weight_value;
        }
        layer->set_weights(weights);

        // read biases data
        Tensor biases({output_size});
        for(size_t j = 0; j < biases.size(); ++j){
            float bias_value = 0.0f;
            file.read(reinterpret_cast<char*>(&bias_value), sizeof(float));
            biases[j] = bias_value;
        }
        layer->set_biases(biases);

        layers_.push_back(std::move(layer));
    }
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
