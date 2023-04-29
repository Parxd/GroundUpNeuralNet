#ifndef GROUNDUPNEURALNET_CONTAINER_H
#define GROUNDUPNEURALNET_CONTAINER_H

#include <memory>
#include <iostream>
#include <vector>
#include <initializer_list>
#include "../layers/BaseModule.h"
#include "../layers/Linear.h"

class Container
{
public:
    Container() = default;

    ~Container() = default;

    /*
     * @brief Variadic constructor
     * @param layer - Variable number of std::unique BaseModule pointers
     * @return Container object with layers added to internal vector
     */
    template<typename... T>
    explicit Container(std::unique_ptr<T>&&... layer)
    {
        (mLayers.push_back(std::move(layer)), ...);
    }

    /*
     * @brief Prints order and description of each layer in internal vector
     */
    void view();

private:
    std::vector<std::unique_ptr<BaseModule>> mLayers;
};

#endif //GROUNDUPNEURALNET_CONTAINER_H
