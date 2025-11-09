#pragma once
#include "layer.hpp"
#include <vector>
#include <memory>

namespace neuralcpp {

class NeuralNetwork {
public:
    /**
     * @brief Default constructor for the neural_network class, 
     * initializes an empty neural network with no layers.
     * @note Please initialize the your neural networks with the createNeuralNetwork()
     * factory function to ensure proper memory management.
     * @see createNeuralNetwork
    */
    NeuralNetwork();

    /**
     * @brief Appends a layer to the network, transferring ownership.
     * Transfers ownership of the provided std::unique_ptr<Layer> into the
     * internal layers_ container. After this call the caller's unique_ptr will
     * no longer own the object (it will be null).
     * @param[in] layer  A std::unique_ptr<Layer> owning the layer to append.
     * @see Layer
     */
    void addLayer(std::unique_ptr<Layer> layer);

    /**
     * @brief forward perfoms a foward pass through the network, 
     * then cashes the output for potential later use.
     * 
     * @param input The input tensor to the neural network.
     * 
     * @return The output tensor from the neural network after the forward pass.
     * 
     * @note for more information on forward passes, see each Layer's forward() method.
     */
    Tensor forward(const Tensor& input);

    /**
     * @brief backward performs backpropagation through the network, 
     * updating weights based on the provided target and learning rate.
     * 
     * @param target The target tensor used to compute the loss gradient.
     * @param learning_rate The learning rate for weight updates.
     * 
     * @note This method assumes that the last forward pass's output is cached.
     * @note for more information on backpropagation, see each Layer's backward() method.
     * 
     */
    void backward(const Tensor& target, float learning_rate);

    /**
     * @brief Getter returns the number of layers in the network.
     * 
     * @return size_t The number of layers in the neural network.
     */
    size_t getNumLayers() const { return layers_.size(); };

    /**
     * @brief Getter returns a const reference to a specific layer in the network.
     * 
     * @param index The index of the layer to retrieve.
     * 
     * @return A const reference to the Layer at the specified index.
     */
    const Layer& getLayer(size_t index) const {return *layers_[index]; };

    /** 
     * @brief Saves the model archetecture, parameters, and functions to a binary file.
     * @param filepath The path to the file where the model will be saved.
     * @throws std::runtime_error if the file cannot be opened for writing.
     */
    void saveModel(const std::string& filepath) const;

    /** 
     * @brief Loads and reconstructs the model from a saved binary file.
     * @param filepath The path to the file where the model will be saved.
     * @throws std::runtime_error if the file cannot be opened for writing.
     * @note This will clear any existing layers in the network before loading.
     */
    void loadModel(const std::string& filepath);

private: 
    std::vector<std::unique_ptr<Layer>> layers_;
    Tensor last_output_;
};

// ----- Factory Functions -----
/**
 * @brief Factor function to create an instance of the NeuralNetwork class.
 * 
 * @return A std::unique_ptr<NeuralNetwork> owning the created empty neural network.
 */
std::unique_ptr<NeuralNetwork> createNeuralNetwork();

/**
 * @brief Factory function to create a DenseLayer with specified parameters.
 * 
 * @param input_size The size of the input to the dense layer.
 * @param output_size The size of the output from the dense layer.
 * @param activation The activation function to use in the dense layer.
 * @param weight_init The weight initialization strategy.
 * @param bias_init The bias initialization strategy.
 * 
 * @return A std::unique_ptr<DenseLayer> owning the created dense layer.
 * 
 * @note Please make sure for layers input and output sizes to match surrounding layers.
 */
std::unique_ptr<DenseLayer> createDenseLayer(size_t input_size, size_t output_size, Activation activation, WeightInitialization weight_init, BiasInitialization bias_init);


} // namespace neuralcpp