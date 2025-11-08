#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include "../inc/dataset.hpp"
#include "../inc/tensor.hpp"
#include "../inc/neural_network.hpp"
#include "../inc/math_ops.hpp"
#include "../inc/layer.hpp"

using namespace neuralcpp;

int main(){
    Dataset dataset;
    dataset.import_loan_data("../datasets/loan_predictions/loan_data.csv", 0.8);
    const auto& training_samples = dataset.getTrainingSamples();
    const auto& testing_samples = dataset.getTestingSamples();
    std::cout << "Training samples loaded: " << training_samples.size() << "\n";
    std::cout << "Testing samples loaded: " << testing_samples.size() << "\n";
    // return 0;
    const size_t epochs = 10;
    const float learning_rate = 0.01f;
    std::mt19937 rng(42);

    auto net = createNeuralNetwork();
    net->addLayer(createDenseLayer(11, 16, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(16, 8, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(8, 1, Activation::Sigmoid, WeightInitialization::Xavier, BiasInitialization::Zero));

    for(size_t e = 0; e < epochs; ++e){
        std::cout << "Started epoch " << (e+1) << "\n";
        double epoch_loss = 0.0;

        for(size_t idx = 0; idx < training_samples.size(); ++idx){
            const Sample& s = training_samples[idx];

            net->forward(s.input);
            net->backward(s.target, learning_rate);
            // compute loss (using available MSE interface since backward uses MSE derivative)
            float loss = math::mse_loss(net->forward(s.input), s.target);
            epoch_loss += loss;
        }
        std::cout << "Epoch " << (e+1) << " completed. Average Loss: " << (epoch_loss / training_samples.size()) << "\n";
    }

    // Evaluate on testing set
    size_t correct_predictions = 0;
    for(const auto& s : testing_samples){
        Tensor output = net->forward(s.input);
        float predicted = output[0] >= 0.5f ? 1.0f : 0.0f;
        if(predicted == s.target[0]){
            ++correct_predictions;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / testing_samples.size();
    std::cout << "Testing Accuracy: " << (accuracy * 100.0f) << "%\n";

    return 0;
}