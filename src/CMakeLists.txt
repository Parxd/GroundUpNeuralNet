target_sources(Core PRIVATE
        ./layers/Linear.cpp
        ./layers/RELU.cpp
        ./layers/Sigmoid.cpp
        ./layers/Softmax.cpp

        ./losses/CE.cpp
        ./losses/MSE.cpp
        )