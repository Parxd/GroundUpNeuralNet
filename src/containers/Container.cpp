#include "../../include/containers/Container.h"
#include "../../include/losses/MSE.h"

void Container::view() {
    int c;
    for (auto &i : mLayers) {
        std::cout << c << ". " << i->getName() << " layer";
        // RTTI check just for the Linear class (special description)
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

void Container::backward(const Eigen::MatrixXf& pred, const Eigen::MatrixXf& target) {
    MSE::forward(pred, target);
    auto errorDerivative = MSE::backward(pred, target);

    auto output = mLayers.back()->backward(errorDerivative);
    for (auto it = mLayers.rbegin() + 1; it != mLayers.rend(); ++it) {
        output = (*it)->backward(output);
    }
}
