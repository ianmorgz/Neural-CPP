/*
    * Tensor class for multi-dimensional arrays.
    * Supports basic operations like copying, indexing, filling, and randomization.
    * Ian Morgan 10-20-25
*/

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

namespace neuralcpp {

class Tensor{
private: 
    std::vector<size_t> shape_; // Dimension of the tensor
    std::vector<float> data_; // flat array of the tensor's data
    size_t calculate_flat_index(const std::vector<size_t>& indices) const;

public:
    // --- Constructors  ---
    Tensor() = default;
    Tensor(const std:: vector<size_t>& shape);
    Tensor(const std:: vector<size_t>& shape, const std:: vector<float>& data);

    // --- Copy and Move Functions ---
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // --- Accessors ---
    float& operator[](size_t index){ return data_[index]; };
    const float& operator[](size_t index) const{ return data_[index]; };

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // --- Getters ---
    const std::vector<size_t>& shape() const{ return shape_; };
    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.size(); }

    // --- Operations ---
    void fill(float value);
    void randomize(float lower=-1.0f, float upper=1.0f);
    void normalize(float lower=0.0f, float upper=1.0f);
    Tensor flatten() const;
};

}; // namespace neuralcpp