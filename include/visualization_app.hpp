#pragma once

#include "math.hpp"

#include <memory>
#include <string>

struct GLFWwindow;
struct cudaGraphicsResource;

namespace viz {

class VisualizationApp {
public:
    VisualizationApp(int width, int height, std::string title);
    ~VisualizationApp();

    VisualizationApp(const VisualizationApp&) = delete;
    VisualizationApp& operator=(const VisualizationApp&) = delete;

    void run();

private:
    void initWindow();
    void initOpenGLResources();
    void initCudaResources();
    void destroyCudaResources();
    void destroyOpenGLResources();

    void update(float deltaSeconds);
    void draw();
    void drawHeatmap();
    void drawPointCloud();

    void compileShaders();
    unsigned int compileShader(unsigned int type, const char* source);
    unsigned int linkProgram(unsigned int vertexShader, unsigned int fragmentShader);

    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::string title_;

    unsigned int heatmapProgram_ = 0;
    unsigned int pointProgram_ = 0;
    unsigned int quadVao_ = 0;
    unsigned int quadVbo_ = 0;
    unsigned int pointVao_ = 0;
    unsigned int pointVbo_ = 0;
    unsigned int texture_ = 0;
    unsigned int pbo_ = 0;

    cudaGraphicsResource* cudaPboResource_ = nullptr;
    cudaGraphicsResource* cudaVboResource_ = nullptr;

    float elapsedTime_ = 0.0f;

    static constexpr int kPointCount = 65536;
};

} // namespace viz
