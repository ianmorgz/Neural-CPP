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

int main(){
    using namespace neuralcpp;

    // adjust path if running from project root or build dir
    Dataset dataset("../datasets/mnist_handwriting/ubyte/");
    dataset.import_training_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    const auto& samples = dataset.getTrainingSamples();
    if(samples.empty()){
        std::cerr << "No training samples loaded. Check dataset path." << std::endl;
        return 1;
    }

    // build a simple feed-forward network: 784 -> 128 -> 10
    auto net = createNeuralNetwork();
    net->addLayer(createDenseLayer(28*28, 128, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(128, 10, Activation::Softmax, WeightInitialization::Xavier, BiasInitialization::Zero));

    
    std::cout << "Starting training on MNIST dataset with " << samples.size() << " samples.\n";

    const size_t epochs = 1;
    const float learning_rate = 0.01f;
    std::mt19937 rng(42);

    // indices for shuffling
    std::vector<size_t> indices(samples.size()-10); // leave some out for quick validation
    for(size_t i = 0; i < indices.size(); ++i) indices[i] = i;


    for(size_t e = 0; e < epochs; ++e){
        std::cout << "Started epoch " << (e+1) << "\n";
        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;

        for(size_t idx = 0; idx < indices.size(); ++idx){
            if(idx % 500 == 0) std::cout << "Processing sample " << idx << "/" << indices.size() << "    | loss: " << (epoch_loss / (idx + 1)) << "\n";
            const Sample& s = samples[indices[idx]];

            // forward
            Tensor out = net->forward(s.input);

            // compute loss (using available MSE interface since backward uses MSE derivative)
            float loss = math::mse_loss(out, s.target);
            epoch_loss += loss;

            // backward (in-place update inside layers)
            net->backward(s.target, learning_rate);
        }

        std::cout << "Epoch " << (e+1) << " / " << epochs << ": avg loss = " << (epoch_loss / samples.size()) << std::endl;
    }

    for(int i = samples.size()-10; i < samples.size(); ++i){
        const Sample& s = samples[i];
        Tensor out = net->forward(s.input);

        // find predicted class
        size_t predicted = 0;
        float max_val = out[0];
        for(size_t j = 1; j < out.size(); ++j){
            if(out[j] > max_val){
                max_val = out[j];
                predicted = j;
            }
        }

        // find actual class
        size_t actual = 0;
        for(size_t j = 0; j < s.target.size(); ++j){
            if(s.target[j] == 1.0f){
                actual = j;
                break;
            }
        }

        std::cout << "Sample " << i << ": Predicted = " << predicted << ", Actual = " << actual << std::endl;
    }

    // save the trained model
    // try{
    //     net->saveModel("mnist_model.bin");
    //     std::cout << "Saved model to mnist_model.bin" << std::endl;
    // } catch(const std::exception& ex){
    //     std::cerr << "Warning: failed to save model: " << ex.what() << std::endl;
    // }

    return 0;
}