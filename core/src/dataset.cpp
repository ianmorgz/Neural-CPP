#include "../inc/dataset.hpp"
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdint>

namespace neuralcpp{

int getHeaderValue(std::ifstream& file, uint16_t offset, uint16_t size){
    file.seekg(offset);
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), size);
    value = __builtin_bswap32(value);
    return value;
}

void normalizeVector(std::vector<float>& vec){
    if(vec.empty()) return; // nothing to normalize
    float max_val = *std::max_element(vec.begin(), vec.end());
    float min_val = *std::min_element(vec.begin(), vec.end());
    if(max_val == min_val){
        // all values are the same; set them to 0.0 to avoid division by zero
        for(auto& v : vec) v = 0.0f;
        return;
    }
    for(auto& v : vec){
        v = (v - min_val) / (max_val - min_val);
    }
}

void Dataset::import_mnist_data(std::string images_path, std::string labels_path, double train_split){
    // cosntruct the full paths for the dataset files

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

void Dataset::import_loan_data(std::string data_path, double train_split){
    std::ifstream load_csv(data_path);
    if(!load_csv.is_open()){
        std::__throw_failure("Error opening loan data file.");
        return;
    }

    std::cerr << "[Dataset] Opened loan CSV: " << data_path << "\n";

    std::vector<Sample> data;
    std::vector<float> gender;
    std::vector<float> married;
    std::vector<float> dependents;
    std::vector<float> education;
    std::vector<float> self_employed;
    std::vector<float> applicant_income;
    std::vector<float> coapplicant_income;
    std::vector<float> loan_amount;
    std::vector<float> loan_amount_term;
    std::vector<float> credit_history;
    std::vector<float> property_area;
    std::vector<float> loan_status;
    std::string line;

    // skip header
    std::getline(load_csv, line);

    while(std::getline(load_csv, line)){
        std::stringstream ss(line);
        std::string value;

        //skip customer id
        getline(ss, value, ',');

        // Gender
        getline(ss, value, ','); 
        gender.push_back( value == "Male" ? 1.0f : 0.0f );

         // Married
        getline(ss, value, ',');
        married.push_back( value == "Yes" ? 1.0f : 0.0f );

        // Dependents
        getline(ss, value, ',');
        dependents.push_back( std::stof(value) );

        // Education
        getline(ss, value, ',');
        education.push_back( value == "Graduate" ? 1.0f : 0.0f );

        // Self_Employed
        getline(ss, value, ',');
        self_employed.push_back( value == "Yes" ? 1.0f : 0.0f );

        // Applicant_Income
        getline(ss, value, ',');
        applicant_income.push_back( std::stof(value) );

        // Coapplicant_Income
        getline(ss, value, ',');
        coapplicant_income.push_back( std::stof(value) );

        // Loan_Amount
        getline(ss, value, ',');
        loan_amount.push_back( std::stof(value) );

        // Loan_Amount_Term
        getline(ss, value, ',');
        loan_amount_term.push_back( std::stof(value) );

        // Credit_History
        getline(ss, value, ',');
        credit_history.push_back( std::stof(value) );
        // Property_Area
        getline(ss, value, ',');
        if(value == "Urban"){
            property_area.push_back(2.0f);
        } else if(value == "Semiurban"){
            property_area.push_back(1.0f);
        } else {
            property_area.push_back(0.0f);
        }

        // Loan_Status
        getline(ss, value, ',');
        loan_status.push_back( value == "Y" ? 1.0f : 0.0f );   
    }

    std::cerr << "[Dataset] Parsed CSV rows: " << gender.size() << "\n";

    // normalize and create tensors
    std::cerr << "[Dataset] Normalizing vectors...\n";
    normalizeVector(applicant_income);
    normalizeVector(coapplicant_income);
    normalizeVector(loan_amount);
    normalizeVector(loan_amount_term);
    normalizeVector(credit_history);
    normalizeVector(property_area);
    std::cerr << "[Dataset] Normalization complete.\n";

    
    for(int i = 0; i < gender.size(); ++i){
        Tensor input({11});
        input[0] = gender[i];
        input[1] = married[i];
        input[2] = dependents[i];
        input[3] = education[i];
        input[4] = self_employed[i];
        input[5] = applicant_income[i];
        input[6] = coapplicant_income[i];
        input[7] = loan_amount[i];
        input[8] = loan_amount_term[i];
        input[9] = credit_history[i];
        input[10] = property_area[i];
        Tensor target({1});
        target[0] = loan_status[i];
        Sample sample;
        sample.input = input;
        sample.target = target;
        data.push_back(sample);
    }

    // split into training and testing sets
    size_t train_size = static_cast<size_t>(data.size() * train_split);
    std::cerr << "[Dataset] Splitting data: total=" << data.size() << " train_size=" << train_size << "\n";
    training_dataset_.assign(data.begin(), data.begin() + train_size);
    testing_dataset_.assign(data.begin() + train_size, data.end());
    std::cout << "Imported " << training_dataset_.size() << " training samples and " 
              << testing_dataset_.size() << " testing samples successfully.\n";
    load_csv.close();
}
} // namespace neuralcpp