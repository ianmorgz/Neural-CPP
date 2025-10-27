#include "../inc/neural_network.hpp"
#include "../inc/layer.hpp"
#include "../inc/math_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// define the training date for the parity example
struct TrainingData {
    std::vector<float> input;
    std::vector<float> target;
};

int main(){
    std::cout << "Starting Parity example...\n";

    // create the parity data set
    // define as a 3 bit parity, returns 1 if there are an odd number of 1s in the input
    // returns 0 if there are an even number of 1s in the input
    // note i removed the following input to test a trained model on data it has never seen
    //    {{1.0f, 0.0f, 1.0f}, {0.0f}},
    std::vector<TrainingData> training_data = {
        {{0.0f, 0.0f, 0.0f}, {0.0f}},
        {{0.0f, 1.0f, 0.0f}, {1.0f}},
        {{0.0f, 1.0f, 1.0f}, {0.0f}},
        {{1.0f, 0.0f, 0.0f}, {1.0f}},
        {{1.0f, 1.0f, 0.0f}, {0.0f}},
        {{1.0f, 1.0f, 1.0f}, {1.0f}}
    };

    std::vector<TrainingData> test_data = {
        {{1.0f, 0.0f, 1.0f}, {0.0f}},
        {{0.0f, 0.0f, 1.0f}, {1.0f}},
    };

    // create the neural network
    auto network = neuralcpp::createNeuralNetwork();

    // add the hidden layers
    network->addLayer(neuralcpp::createDenseLayer(3, 16, neuralcpp::Activation::Tanh, neuralcpp::WeightInitialization::Xavier, neuralcpp::BiasInitialization::Uniform));

    network->addLayer(neuralcpp::createDenseLayer(16, 8, neuralcpp::Activation::Tanh, neuralcpp::WeightInitialization::Xavier, neuralcpp::BiasInitialization::Uniform));

    // add the output layer
    network->addLayer(neuralcpp::createDenseLayer(8, 1, neuralcpp::Activation::Sigmoid, neuralcpp::WeightInitialization::Xavier, neuralcpp::BiasInitialization::Uniform));

    std::cout << "Network created with 2 layers:\n";
    std::cout << "- Input: 3 neurons\n";
    std::cout << "- Hidden: 16 neurons (Tanh)\n";
    std::cout << "- Hidden: 8 neurons (Tanh)\n";
    std::cout << "- Output: 1 neuron (Sigmoid)\n";
    std::cout << "\nStarting training...\n";

    const int epochs = 10000;
    const float learning_rate = 0.1f;
    for(int epoch = 0; epoch < epochs; epoch++){
        float current_learning_rate = learning_rate * (1.0f - (float)epoch / epochs);
        float total_loss = 0.0f;
        for(const auto& data : training_data){
            // prepare input tensor
            neuralcpp::Tensor input({3});
            input[0] = data.input[0];
            input[1] = data.input[1];
            input[2] = data.input[2];

            // prepare target tensor
            neuralcpp::Tensor target({1});
            target[0] = data.target[0];

            // forward pass
            neuralcpp::Tensor output = network->forward(input);

            // compute loss
            float loss = neuralcpp::math::mse_loss(output, target);
            total_loss += loss;

            // backward pass
            network->backward(target, current_learning_rate);

        }
        if(epoch % 500 == 0){
            std::cout << "Epoch " << epoch << "/" << epochs << " - Loss: " << total_loss / training_data.size() << "\n";
        }
    }
    std::cout << "\nTraining completed!\n";
    std::cout << "\nTesting the trained network:\n";

    // Test the trained network
    for (const auto& data : training_data) {
        neuralcpp::Tensor input({3});
        input[0] = data.input[0];
        input[1] = data.input[1];
        input[2] = data.input[2];
        
        auto output = network->forward(input);
        
        std::cout << "Input: [" << data.input[0] << ", " << data.input[1]<< ", " << data.input[2] << "] -> ";
        std::cout << "Output: " << output[0] << " (expected: " << data.target[0] << ")\n";
    }

    std:: cout << "\nTesting on unseen data:\n";
    for (const auto& data : test_data) {
        neuralcpp::Tensor input({3});
        input[0] = data.input[0];
        input[1] = data.input[1];
        input[2] = data.input[2];

        auto output = network->forward(input);

        std::cout << "Input: [" << data.input[0] << ", " << data.input[1]<< ", " << data.input[2] << "] -> ";
        std::cout << "Output: " << output[0] << " (expected: " << data.target[0] << ")\n";
    }
    return 0;
}