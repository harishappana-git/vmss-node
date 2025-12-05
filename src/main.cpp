#include "visualization_app.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        const int width = 1600;
        const int height = 900;
        viz::VisualizationApp app(width, height, "CUDA Visual Analytics");
        app.run();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
