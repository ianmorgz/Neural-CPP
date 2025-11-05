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
    Dataset(std::string base_path)
        : dataset_base_path_(base_path) {};

    void import_training_data(std::string images_filename, std::string label_filename);
    void import_testing_data(std::string images_filename, std::string label_filename);
    //TODO: redo this function to read from ubyte files directly

    // Read-only accessors for loaded samples so examples and trainers can iterate over data
    const std::vector<Sample>& getTrainingSamples() const { return training_dataset_; }
    const std::vector<Sample>& getTestingSamples() const { return testing_dataset_; }

private:
    int getHeaderValue(std::ifstream& file, uint16_t offset, uint16_t size);
    // paths to the dataset files
    std::string dataset_base_path_;

    // storage for the dataset samples
    std::vector<Sample> training_dataset_;
    std::vector<Sample> testing_dataset_;

};
} // namespace neuralcpp