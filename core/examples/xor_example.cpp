#include "../inc/neural_network.hpp"
#include "../inc/layer.hpp"
#include "../inc/math_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>


// define the training data for the xor example
struct TrainingData {
    std::vector<float> input;
    std::vector<float> target;
};

int main(){
    std::cout << "Starting XOR example...\n";

    // Create training data for XOR (exclusive OR)
    // XOR truth table:
    // 0,0 -> 0
    // 0,1 -> 1  
    // 1,0 -> 1
    // 1,1 -> 0


    std::vector<TrainingData> training_data = {
        {{0.0f, 0.0f}, {0.0f}},
        {{0.0f, 1.0f}, {1.0f}},
        {{1.0f, 0.0f}, {1.0f}},
        {{1.0f, 1.0f}, {0.0f}}
    };
    
    // create the neural network
    auto network = neuralcpp::createNeuralNetwork();

    // add two layers:
    // the hidden layer: 
    network->addLayer(neuralcpp::createDenseLayer(2, 4, neuralcpp::Activation::ReLU));
    // the output layer:
    network->addLayer(neuralcpp::createDenseLayer(4, 1, neuralcpp::Activation::Sigmoid));

    std::cout << "Network created with 2 layers:\n";
    std::cout << "- Input: 2 neurons\n";
    std::cout << "- Hidden: 4 neurons (ReLU)\n";
    std::cout << "- Output: 1 neuron (Sigmoid)\n";
    std::cout << "\nStarting training...\n";

    // training loop
    const int epochs = 5000;
    const float learning_rate = 0.1f;
    for(int epoch = 0; epoch < epochs; epoch++){
        float total_loss = 0.0f;
        for(const auto& data : training_data){
            // prepare input tensor
            neuralcpp::Tensor input({2});
            input[0] = data.input[0];
            input[1] = data.input[1];

            // prepare target tensor
            neuralcpp::Tensor target({1});
            target[0] = data.target[0];

            // perform a forward pass
            auto output = network->forward(input);

            // compute loss with MSE function
            float loss = neuralcpp::math::mse_loss(output, target);
            total_loss += loss;

            // perform backward pass
            network->backward(target, learning_rate);
        }

        if(epoch % 500 == 0){
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / training_data.size() << "\n";
        }

        std::cout << "\nTraining completed!\n";
        std::cout << "\nTesting the trained network:\n";

        // Test the trained network
        for (const auto& data : training_data) {
            neuralcpp::Tensor input({2});
            input[0] = data.input[0];
            input[1] = data.input[1];
            
            auto output = network->forward(input);
            
            std::cout << "Input: [" << data.input[0] << ", " << data.input[1] << "] -> ";
            std::cout << "Output: " << output[0] << " (expected: " << data.target[0] << ")\n";
        }
        
        return 0;
    }


}