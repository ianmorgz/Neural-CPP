#pragma once
#include "layer.hpp"
#include <vector>
#include <memory>

namespace neuralcpp{
class NeuralNetwork {
public:
    NeuralNetwork();

    void addLayer(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& target, float learning_rate);

    size_t getNumLayers() const { return layers_.size(); };
    const Layer& getLayer(size_t index) const {return *layers_[index]; };

    void saveModel(const std::string& filepath) const;
    void loadModel(const std::string& filepath);
private: 
    std::vector<std::unique_ptr<Layer>> layers_;
    Tensor last_output_;
};

std::unique_ptr<NeuralNetwork> createNeuralNetwork();
std::unique_ptr<DenseLayer> createDenseLayer(size_t input_size, size_t output_size, Activation activation, WeightInitialization weight_init, BiasInitialization bias_init);


} // namespace neuralcpp