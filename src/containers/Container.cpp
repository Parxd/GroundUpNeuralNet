#include "../../include/containers/Container.h"

template<typename ...T>
Container::Container(T *...layer) {
    (mLayers.push_back(std::unique_ptr<BaseModule>(layer)), ...);
}

void Container::view() {
    int c = 0;
    for (auto &i : mLayers) {
        std::cout << c << ". " << i->getName() << " layer";

        // RTTI for Linear class check
        if (dynamic_cast<Linear *>(i.get())) {
            std::cout << " with " << i->getInputs() << " inputs and " << i->getOutputs() << " outputs";
        }
        std::cout << "\n";
        ++c;
        std::flush(std::cout);
    }
}