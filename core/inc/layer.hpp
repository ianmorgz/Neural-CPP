#pragma once
#include "tensor.hpp"
#include <memory>
#include <functional>

namespace neuralcpp {

enum class Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    // Linear
};

class Layer{
public:
    virtual ~Layer() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_ouput, float learning_rate) = 0;

    virtual const Tensor& get_weights() const = 0;
    virtual const Tensor& get_biases() const = 0;

    virtual void set_weights(const Tensor& weights) = 0;
    virtual void set_biases(const Tensor& biases) = 0;
};

class DenseLayer : public Layer {
private:
    Tensor weights_;
    Tensor biases_;
    Tensor input_cache_; // Cache input for use in backpropagation
    Tensor output_cache_; // Cache output for use in backpropagation
    Activation activation_;

    float activation_function(float x) const;
    float activation_derivative(float x) const;
    void apply_activation(Tensor& tensor) const;
    void apply_activation_derivative(Tensor& tensor) const;

public:

    DenseLayer(size_t input_size, size_t output_size, Activation activation);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& get_weights() const override { return weights_; }
    const Tensor& get_biases() const override { return biases_; }

    void set_weights(const Tensor& weights) override;
    void set_biases(const Tensor& biases) override;
};

} // namespace neuralcpp