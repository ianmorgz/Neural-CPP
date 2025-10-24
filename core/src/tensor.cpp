#include "../inc/tensor.hpp"
#include <random>
#include <stdexcept>

namespace neuralcpp {
// --- Private Functions ---
size_t Tensor::calculate_flat_index(const std::vector<size_t>& indices) const {
    if(indices.size() != shape_.size()){
        throw std::invalid_argument("Number of indices does not match tensor dimensions.");
    }

    size_t index = 0;
    size_t stride = 1;

    for(int i = shape_.size() - 1; i >= 0; --i){
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return index;
};

// --- Constructors ---
Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    // calculate total size of the tensor
    size_t total_size = 1;
    for (const auto& dim : shape) {
        total_size *= dim;
    }
    // Initialize data with zeros
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const std:: vector<size_t>& shape, const std:: vector<float>& data) : shape_(shape), data_(data){
    size_t total_size = 1;
    for (const auto& dim : shape) {
        total_size *= dim;
    }
    if (data.size() != total_size) {
        throw std::invalid_argument("Data size does not match tensor shape.");
    }
}

// --- Copy and Move Functions ---
Tensor::Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_) {};
Tensor::Tensor(Tensor&& other) noexcept : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {};
Tensor& Tensor::operator=(const Tensor& other) { 
    if(this != &other){
        shape_ = other.shape_;
        data_ = other.data_;
    }
    return *this;
};

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}

// -- Acsessors ---
float& Tensor::at(const std::vector<size_t>& indices) {
    size_t flat_index = calculate_flat_index(indices);
    return data_.at(flat_index);
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t flat_index = calculate_flat_index(indices);
    return data_.at(flat_index);
}

// --- Operations ---
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}
void Tensor::randomize(float lower, float upper) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower, upper);
    
    for (float& val : data_) {
        val = dis(gen);
    }
}

Tensor Tensor::flatten() const {
    return Tensor({data_.size()}, data_);
}

}; // namespace neuralcpp