#include "../inc/dataset.hpp"
#include <random>

namespace neuralcpp{

size_t Dataset::getInputSize() const {
    if(training_dataset_.empty()){
        throw std::runtime_error("Training dataset is empty. Cannot determine input size.");
    }
    return training_dataset_[0].input.size();
}

size_t Dataset::getOutputSize() const {
    if(training_dataset_.empty()){
        throw std::runtime_error("Training dataset is empty. Cannot determine output size.");
    }
    return training_dataset_[0].target.size();
}

void Dataset::shuffleTrainingData(){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_dataset_.begin(), training_dataset_.end(), g);
}

void Dataset::shuffleTestingData(){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(testing_dataset_.begin(), testing_dataset_.end(), g);
}

void Dataset::validateTrainSplit(double train_split) const {
    if(train_split <= 0.0 || train_split >= 1.0){
        throw std::invalid_argument("Train split must be between 0 and 1 (exclusive).");
    }
}

void Dataset::splitDataset(const std::vector<Sample>& full_dataset, double train_split) {
    size_t train_size = static_cast<size_t>(full_dataset.size() * train_split);
    training_dataset_.assign(full_dataset.begin(), full_dataset.begin() + train_size);
    testing_dataset_.assign(full_dataset.begin() + train_size, full_dataset.end());
}

std::vector<Sample> Dataset::getTrainingBatch(size_t batch_size, size_t start_index) const {
    std::vector<Sample> batch;
    if(start_index >= training_dataset_.size()){
        return batch; // empty batch
    }
    size_t end_index = std::min(start_index + batch_size, training_dataset_.size());
    batch.insert(batch.end(), training_dataset_.begin() + start_index, training_dataset_.begin() + end_index);
    return batch;
}

std::vector<Sample> Dataset::getTestingBatch(size_t batch_size, size_t start_index) const {
    std::vector<Sample> batch;
    if(start_index >= testing_dataset_.size()){
        return batch; // empty batch
    }
    size_t end_index = std::min(start_index + batch_size, testing_dataset_.size());
    batch.insert(batch.end(), testing_dataset_.begin() + start_index, testing_dataset_.begin() + end_index);
    return batch;
}

} // namespace neuralcpp