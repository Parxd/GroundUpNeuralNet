#ifndef GROUNDUPNEURALNET_CONTAINER_H
#define GROUNDUPNEURALNET_CONTAINER_H

#include <memory>
#include <vector>
#include <initializer_list>
#include "../layers/BaseModule.h"

class Container
{
public:
    Container();
    explicit Container(const std::vector<std::unique_ptr<BaseModule>>&);
private:
    std::vector<std::unique_ptr<BaseModule>> layers;

};

#endif //GROUNDUPNEURALNET_CONTAINER_H
