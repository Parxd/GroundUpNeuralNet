#include <fstream>
#include <filesystem>
#include "../../include/containers/Container.h"

void Container::view() {
    int c;
    for (auto &i : mLayers) {
        std::cout << c << ". " << i->getName() << " layer";
        if (dynamic_cast<Linear *>(i.get())) {
            std::cout << " -> " << i->getInputs() << " inputs // " << i->getOutputs() << " outputs // ";
            std::cout << "η: " << i->getLR();
        }
        std::cout << "\n";
        ++c;
        std::flush(std::cout);
    }
}

Eigen::MatrixXf Container::forward(const Eigen::MatrixXf& input) {
    auto output = mLayers[0]->forward(input);
    for (auto it = mLayers.begin() + 1; it != mLayers.end(); ++it) {
        output = (*it)->forward(output);
    }
    return output;
}

void Container::backward(const Eigen::MatrixXf &errorDerivative) {
    auto output = mLayers.back()->backward(errorDerivative);
    for (auto it = mLayers.rbegin() + 1; it != mLayers.rend(); ++it) {
        output = (*it)->backward(output);
    }
}

void Container::save(const std::string& file, const std::string& name) {
    if (std::filesystem::is_directory(file)) {
        write(file + "/model.csv", name);
        std::cout << "Model successfully saved.";
    }
    else if (std::filesystem::is_regular_file(file)) {
        write(file, name);
        std::cout << "Model successfully saved.";
    }
    else {
        throw std::invalid_argument("Invalid filepath");
    }
}

void Container::write(const std::string &file, const std::string &name) {
    std::ofstream csvFile(file);
    csvFile << name << "\n\n";
    for (auto &i : mLayers) {
        csvFile << i->getName() << "\n";
        auto linearPtr = dynamic_cast<Linear *>(i.get());
        if (linearPtr) {
            csvFile << linearPtr->getWeight() << "\n\n";
            csvFile << linearPtr->getBias() << "\n";
        }
        csvFile << std::endl;
    }
    csvFile.close();
}

void Container::load(const std::string &file) {
    mLayers.clear();

    std::string name;
    int lineCounter = 0;
    std::ifstream csvFile(file);
    std::string line;
    while (std::getline(csvFile, line)) {
        if (!lineCounter) {
            name = line;
        }
        ++lineCounter;
    }

    std::cout << "Model: \"" << name << "\" successfully loaded.";
}

