#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include "../inc/dataset.hpp"
#include "../inc/tensor.hpp"
#include "../inc/neural_network.hpp"
#include "../inc/math_ops.hpp"
#include "../inc/layer.hpp"

class MnistDataset : public neuralcpp::Dataset {
public:
    MnistDataset() = default;

    void import_data(const std::string& data_path, double train_split) override {
        // cosntruct the full paths for the dataset files
        std::string images_path = data_path + "train-images.idx3-ubyte";
        std::string labels_path = data_path + "train-labels.idx1-ubyte";

        // open the files
        std::ifstream images_file(images_path, std::ios::binary);
        std::ifstream labels_file(labels_path, std::ios::binary);

        if (!images_file.is_open() || !labels_file.is_open()) {
            std::__throw_failure("Error opening files.");
            return;
        }

        // read and parse the dataset files
        // starting data - wealredy know this from the MNIST format
        uint16_t MAGIC_OFFSET = 0;
        uint16_t OFFSET_SIZE = 4; //in bytes

        uint16_t LABEL_MAGIC = 2049;
        uint16_t IMAGE_MAGIC = 2051;

        uint16_t NUMBER_ITEMS_OFFSET = 4;
        uint16_t ITEMS_SIZE = 4;

        uint16_t NUMBER_OF_ROWS_OFFSET = 8;
        uint16_t ROWS_SIZE = 4;
        uint16_t ROWS = 28;

        uint16_t NUMBER_OF_COLUMNS_OFFSET = 12;
        uint16_t COLUMNS_SIZE = 4;
        uint16_t COLUMNS = 28;

        uint16_t IMAGE_OFFSET = 16;
        uint16_t IMAGE_SIZE = ROWS * COLUMNS;

        // read headers
        // first start with the magic numbers
        // image file
        uint32_t image_magic_number = getHeaderValue(images_file, MAGIC_OFFSET, OFFSET_SIZE);
        uint32_t label_magic_number = getHeaderValue(labels_file, MAGIC_OFFSET, OFFSET_SIZE);
        if (label_magic_number != LABEL_MAGIC) {
            std::__throw_failure("Invalid label file magic number.");
            return;
        }

        // read number of items
        uint32_t number_of_images = getHeaderValue(images_file, NUMBER_ITEMS_OFFSET, ITEMS_SIZE);
        uint32_t number_of_labels = getHeaderValue(labels_file, NUMBER_ITEMS_OFFSET, ITEMS_SIZE);
        
        // check that the numbers much - saftey net
        if (number_of_images != number_of_labels) {
            std::__throw_failure("Number of images does not match number of labels.");
            return;
        }

        // read the nuber of rows and columns
        uint32_t num_rows = getHeaderValue(images_file, NUMBER_OF_ROWS_OFFSET, ROWS_SIZE);
        uint32_t num_columns = getHeaderValue(images_file, NUMBER_OF_COLUMNS_OFFSET, COLUMNS_SIZE);

        // read the images and labels
        for (uint32_t i = 0; i < number_of_images; ++i) {
            // read the label
            uint8_t label;
            labels_file.read(reinterpret_cast<char*>(&label), 1);
            // read the image
            std::vector<uint8_t> image_data(IMAGE_SIZE);
            images_file.read(reinterpret_cast<char*>(image_data.data()), IMAGE_SIZE);

            // create the input and target tensors
            neuralcpp::Tensor input({num_columns * num_rows});
            for(int i = 0; i < IMAGE_SIZE; ++i){
                input[i] = static_cast<float>(image_data[i]) / 255.0f; // normalize to [0, 1]
            }

            neuralcpp::Tensor target({10}); // one-hot encoding for 10 classes
            target.fill(0.0f);
            target.at({static_cast<size_t>(label)}) = 1.0f;

            // store the sample
            neuralcpp::Sample sample(input, target);

            training_dataset_.push_back(sample);
        }

        std::cout << "Imported " << training_dataset_.size() << " training samples successfully.\n"; 
        // bang out

        // close the files
        images_file.close();
        labels_file.close();
    }

private:
    int getHeaderValue(std::ifstream& file, uint16_t offset, uint16_t size){
        file.seekg(offset);
        uint32_t value = 0;
        file.read(reinterpret_cast<char*>(&value), size);
        value = __builtin_bswap32(value);
        return value;
    }
};

int main(){
    using namespace neuralcpp;

    // adjust path if running from project root or build dir
    MnistDataset dataset;
    dataset.import_data("../datasets/mnist_handwriting/", 0.8);

    const auto& samples = dataset.getTrainingSamples();
    if(samples.empty()){
        std::cerr << "No training samples loaded. Check dataset path." << std::endl;
        return 1;
    }

    // build a simple feed-forward network: 784 -> 128 -> 10
    auto net = createNeuralNetwork();
    net->addLayer(createDenseLayer(28*28, 128, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(128, 10, Activation::Softmax, WeightInitialization::Xavier, BiasInitialization::Zero));

    
    std::cout << "Starting training on MNIST dataset with " << samples.size() << " samples.\n";

    const size_t epochs = 1;
    const float learning_rate = 0.01f;
    std::mt19937 rng(42);

    // indices for shuffling
    std::vector<size_t> indices(samples.size()-10); // leave some out for quick validation
    for(size_t i = 0; i < indices.size(); ++i) indices[i] = i;


    for(size_t e = 0; e < epochs; ++e){
        std::cout << "Started epoch " << (e+1) << "\n";
        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;

        for(size_t idx = 0; idx < indices.size(); ++idx){
            if(idx % 500 == 0) std::cout << "Processing sample " << idx << "/" << indices.size() << "    | loss: " << (epoch_loss / (idx + 1)) << "\n";
            const Sample& s = samples[indices[idx]];

            // forward
            Tensor out = net->forward(s.input);

            // compute loss (using available MSE interface since backward uses MSE derivative)
            float loss = math::mse_loss(out, s.target);
            epoch_loss += loss;

            // backward (in-place update inside layers)
            net->backward(s.target, learning_rate);
        }

        std::cout << "Epoch " << (e+1) << " / " << epochs << ": avg loss = " << (epoch_loss / samples.size()) << std::endl;
    }

    for(int i = samples.size()-10; i < samples.size(); ++i){
        const Sample& s = samples[i];
        Tensor out = net->forward(s.input);

        // find predicted class
        size_t predicted = 0;
        float max_val = out[0];
        for(size_t j = 1; j < out.size(); ++j){
            if(out[j] > max_val){
                max_val = out[j];
                predicted = j;
            }
        }

        // find actual class
        size_t actual = 0;
        for(size_t j = 0; j < s.target.size(); ++j){
            if(s.target[j] == 1.0f){
                actual = j;
                break;
            }
        }

        std::cout << "Sample " << i << ": Predicted = " << predicted << ", Actual = " << actual << std::endl;
    }

    // save the trained model
    // try{
    //     net->saveModel("mnist_model.bin");
    //     std::cout << "Saved model to mnist_model.bin" << std::endl;
    // } catch(const std::exception& ex){
    //     std::cerr << "Warning: failed to save model: " << ex.what() << std::endl;
    // }

    return 0;
}

