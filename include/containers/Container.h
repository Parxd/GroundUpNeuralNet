#ifndef GROUNDUPNEURALNET_CONTAINER_H
#define GROUNDUPNEURALNET_CONTAINER_H

#include <memory>
#include <iostream>
#include <vector>
#include "../layers/BaseModule.h"
#include "../layers/Linear.h"

class Container
{
public:
    Container() = default;

    ~Container() = default;

    /*
     * @brief PREFERRED variadic constructor - accepts std::unique_ptr from BaseModule layer factories
     * @param layer - Variable number of std::unique_ptr of BaseModule type
     * @return Container object with layers added to internal vector
     */
    template<typename... T>
    explicit Container(std::unique_ptr<T>&&... layer)
    {
        (mLayers.push_back(std::move(layer)), ...);
    }

    /*
     * @brief Variadic constructor - converts raw base pointers into std::unique_ptr for trivial destruction
     * @param layer - Variable number of RAW BaseModule pointers
     * @return Container object with layers added to internal vector
     */
    template<typename... T>
    explicit Container(T*... layer)
    {
        (mLayers.push_back(std::unique_ptr<BaseModule>(layer)), ...);
    }

    explicit Container(std::vector<std::unique_ptr<BaseModule>>& layers);

    /*
     * @brief Prints order and description of each layer in internal vector
     */
    void view();

private:
    std::vector<std::unique_ptr<BaseModule>> mLayers;
};

#endif //GROUNDUPNEURALNET_CONTAINER_H
