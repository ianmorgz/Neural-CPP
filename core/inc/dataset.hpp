#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include "tensor.hpp"

namespace neuralcpp {

struct Sample {
    Tensor input;
    Tensor target;
    
    // --- constructors ---
    Sample(Tensor input_tensor, Tensor target_tensor) 
        : input(std::move(input_tensor)), target(std::move(target_tensor)) {}
};

class Dataset {
public:
    Dataset() = default;
    virtual ~Dataset() = default;  // Virtual destructor for proper polymorphism
    
    // Import data with train/test split
    virtual void import_data(const std::string& data_path, double train_split) = 0;
    
    // Batch access for efficient training
    std::vector<Sample> getTrainingBatch(size_t batch_size, size_t start_index = 0) const;
    std::vector<Sample> getTestingBatch(size_t batch_size, size_t start_index = 0) const;
    
    // Accessors
    const std::vector<Sample>& getTrainingSamples() const { return training_dataset_; }
    const std::vector<Sample>& getTestingSamples() const { return testing_dataset_; }
    
    // Dataset statistics
    size_t getTrainingSize() const { return training_dataset_.size(); }
    size_t getTestingSize() const { return testing_dataset_.size(); }
    size_t getInputSize() const; 
    size_t getOutputSize() const;
    
    // Shuffle functionality
    void shuffleTrainingData();
    void shuffleTestingData();

protected:
    std::vector<Sample> training_dataset_;
    std::vector<Sample> testing_dataset_;
    
    // Helper methods for derived classes
    void validateTrainSplit(double train_split) const;
    void splitDataset(const std::vector<Sample>& full_dataset, double train_split);
};

} // namespace neuralcpp