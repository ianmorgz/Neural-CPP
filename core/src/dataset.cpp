#include "../inc/dataset.hpp"
#include <fstream>

namespace neuralcpp{

int Dataset::getHeaderValue(std::ifstream& file, uint16_t offset, uint16_t size){
    file.seekg(offset);
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), size);
    value = __builtin_bswap32(value);
    return value;
}

void Dataset::import_training_data(std::string images_filename, std::string label_filename){
    // cosntruct the full paths for the dataset files
    std::string training_images_path_ = dataset_base_path_ + images_filename;
    std::string training_labels_path_ = dataset_base_path_ + label_filename;

    // open the files
    std::ifstream images_file(training_images_path_, std::ios::binary);
    std::ifstream labels_file(training_labels_path_, std::ios::binary);

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
        Tensor input({num_columns * num_rows});
        for(int i = 0; i < IMAGE_SIZE; ++i){
            input[i] = static_cast<float>(image_data[i]) / 255.0f; // normalize to [0, 1]
        }

        Tensor target({10}); // one-hot encoding for 10 classes
        target.fill(0.0f);
        target.at({static_cast<size_t>(label)}) = 1.0f;

        // store the sample
        Sample sample;
        sample.input = input;
        sample.target = target;

        training_dataset_.push_back(sample);
    }

    std::cout << "Imported " << training_dataset_.size() << " training samples successfully.\n"; 
    // bang out

    // close the files
    images_file.close();
    labels_file.close();

}



} // namespace neuralcpp