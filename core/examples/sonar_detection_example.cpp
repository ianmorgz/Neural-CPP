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

class SonarDataset : public neuralcpp::Dataset {
public:
    SonarDataset() = default;

    void import_data(const std::string& data_path, double train_split) override {
        std::ifstream load_csv(data_path);
        if(!load_csv.is_open()){
            std::__throw_failure("Error opening sonar data file.");
            return;
        }

        std::cerr << "[Dataset] Opened sonar CSV: " << data_path << "\n";

        std::vector<Sample> data;
        std::string line;

        while(std::getline(load_csv, line)){
            std::stringstream ss(line);
            std::string value;
            std::vector<float> input_values;

            // Read 60 input features
            for(int i = 0; i < 60; ++i){
                getline(ss, value, ',');
                input_values.push_back(std::stof(value));
            }

            // Read label
            getline(ss, value, ',');
            Tensor input_tensor({60}, {input_values});
            std::vector<float> target_values = {value == "R" ? 1.0f : 0.0f};
            Tensor target_tensor({1}, {target_values}); // Rock=1.0, Mine=0.0

            data.emplace_back(input_tensor, target_tensor);
        }

        load_csv.close();

        //shuffle the data
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);


        // Split dataset into training and testing sets
        validateTrainSplit(train_split);
        splitDataset(data, train_split);

        std::cerr << "[Dataset] Loaded " << data.size() << " samples. "
                  << training_dataset_.size() << " for training, "
                  << testing_dataset_.size() << " for testing.\n";
    }
};

int main(){
    using namespace neuralcpp;

    SonarDataset dataset;
    dataset.import_data("../datasets/sonar_detection/sonar.csv", 0.9);

    std::cout << "Training samples: " << dataset.getTrainingSize() << "\n";
    std::cout << "Testing samples: " << dataset.getTestingSize() << "\n";

    const size_t epochs = 100;
    const float learning_rate = 0.01f;
    std::mt19937 rng(42);

    auto net = createNeuralNetwork();
    net->addLayer(createDenseLayer(60, 30, Activation::ReLU, WeightInitialization::HeNormal, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(30, 15, Activation::ReLU, WeightInitialization::HeNormal, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(15, 1, Activation::Sigmoid, WeightInitialization::Xavier, BiasInitialization::Zero));

    for(size_t e = 0; e < epochs; ++e){
        dataset.shuffleTrainingData();
        const auto& training_samples = dataset.getTrainingSamples();
        std::cout << "Started epoch " << (e+1) << "\n";
        double epoch_loss = 0.0;

        for(const auto& sample : training_samples){
            Tensor output = net->forward(sample.input);
            epoch_loss += math::mse_loss(output, sample.target);
            net->backward(sample.target, learning_rate);
        }

        epoch_loss /= static_cast<double>(training_samples.size());
        std::cout << "Epoch " << (e+1) << " Loss: " << epoch_loss << "\n";
    }

    // Evaluate on test set
    const auto& testing_samples = dataset.getTestingSamples();
    size_t correct_predictions = 0;
    for(const auto& sample : testing_samples){
        Tensor output = net->forward(sample.input);
        float predicted = output[0] >= 0.5f ? 1.0f : 0.0f;
        if(predicted == sample.target[0]){
            ++correct_predictions;
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / testing_samples.size();
    std::cout << "Testing Accuracy: " << (accuracy * 100.0f) << "%\n";

    return 0;

}