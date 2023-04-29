#include "../../include/containers/Container.h"

void Container::view() {
    int c = 0;
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