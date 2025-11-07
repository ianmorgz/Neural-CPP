#pragma once
#include "tensor.hpp"
#include <memory>
#include <functional>

namespace neuralcpp {

enum class Activation {
    ReLU = 0,
    Sigmoid = 1,
    Tanh = 2,
    Softmax = 3,
    // Linear
};

enum class WeightInitialization {
    Zero,
    Xavier,
    XavierNormal,
    HeUniform,
    HeNormal
};

enum class BiasInitialization {
    Zero,
    Constant,
    Uniform,
    SmartOutput  // Special case for output layers
};

class Layer{
public:
    virtual ~Layer() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_ouput, float learning_rate) = 0;

    virtual const Tensor& get_weights() const = 0;
    virtual const Tensor& get_biases() const = 0;
    virtual const int get_activation_type() const = 0;

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

    void apply_activation(Tensor& tensor) const;
    void apply_activation_derivative(Tensor& tensor) const;

    void initialize_weights(WeightInitialization weight_init);
    void initialize_bias(BiasInitialization bias_init);


public:

    DenseLayer(size_t input_size, size_t output_size, Activation activation, WeightInitialization weight_init, BiasInitialization bias_init);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& get_weights() const override { return weights_; }
    const Tensor& get_biases() const override { return biases_; }
    const int get_activation_type() const override { return static_cast<int>(activation_); }

    void set_weights(const Tensor& weights) override;
    void set_biases(const Tensor& biases) override;
};

} // namespace neuralcpp