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

using namespace neuralcpp;

class LoanDataset : public neuralcpp::Dataset {
public:
    LoanDataset() = default;
    
    void import_data(const std::string& data_path, double train_split) override {
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
            Sample sample(input, target);
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

};


int main(){
    // LoanDataset dataset;
    // dataset.import_loan_data("../datasets/loan_predictions/loan_data.csv", 0.8);

    LoanDataset dataset;
    dataset.import_data("../datasets/loan_predictions/loan_data.csv", 0.8);

    const auto& training_samples = dataset.getTrainingSamples();
    const auto& testing_samples = dataset.getTestingSamples();
    std::cout << "Training samples loaded: " << training_samples.size() << "\n";
    std::cout << "Testing samples loaded: " << testing_samples.size() << "\n";
    // return 0;
    const size_t epochs = 10;
    const float learning_rate = 0.01f;
    std::mt19937 rng(42);

    auto net = createNeuralNetwork();
    net->addLayer(createDenseLayer(11, 16, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(16, 8, Activation::ReLU, WeightInitialization::Xavier, BiasInitialization::Zero));
    net->addLayer(createDenseLayer(8, 1, Activation::Sigmoid, WeightInitialization::Xavier, BiasInitialization::Zero));

    for(size_t e = 0; e < epochs; ++e){
        dataset.shuffleTrainingData();
        const auto& training_samples = dataset.getTrainingSamples();
        std::cout << "Started epoch " << (e+1) << "\n";
        double epoch_loss = 0.0;

        for(size_t idx = 0; idx < training_samples.size(); ++idx){
            const Sample& s = training_samples[idx];

            net->forward(s.input);
            net->backward(s.target, learning_rate);
            // compute loss (using available MSE interface since backward uses MSE derivative)
            float loss = math::mse_loss(net->forward(s.input), s.target);
            epoch_loss += loss;
        }
        std::cout << "Epoch " << (e+1) << " completed. Average Loss: " << (epoch_loss / training_samples.size()) << "\n";
    }

    // Evaluate on testing set
    size_t correct_predictions = 0;
    for(const auto& s : testing_samples){
        Tensor output = net->forward(s.input);
        float predicted = output[0] >= 0.5f ? 1.0f : 0.0f;
        if(predicted == s.target[0]){
            ++correct_predictions;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / testing_samples.size();
    std::cout << "Testing Accuracy: " << (accuracy * 100.0f) << "%\n";

    return 0;
}