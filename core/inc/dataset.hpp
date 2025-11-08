#pragma once

#include <vector>
#include <iostream>
#include "tensor.hpp"

namespace neuralcpp {

struct Sample {
    Tensor input;
    Tensor target;
};
// https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
class Dataset {
public:
    // constructor
    Dataset() = default;

    void import_mnist_data(std::string images_path, std::string labels_path, double train_split);
    void import_loan_data(std::string data_path, double train_split);
 
    const std::vector<Sample>& getTrainingSamples() const { return training_dataset_; }
    const std::vector<Sample>& getTestingSamples() const { return testing_dataset_; }

private:

    // storage for the dataset samples
    std::vector<Sample> training_dataset_;
    std::vector<Sample> testing_dataset_;

};
} // namespace neuralcpp