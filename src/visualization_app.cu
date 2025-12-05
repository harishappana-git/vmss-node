#include "visualization_app.hpp"

#include "cuda_kernels.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace viz {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << '\n';
}

inline void throwIfCudaFailed(cudaError_t error, const char* action)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
    }
}

void checkOpenGlError(const char* action)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        std::ostringstream oss;
        oss << std::hex << "0x" << error;
        throw std::runtime_error(std::string(action) + ": OpenGL error " + oss.str());
    }
}

} // namespace

VisualizationApp::VisualizationApp(int width, int height, std::string title)
    : width_(width), height_(height), title_(std::move(title))
{
}

VisualizationApp::~VisualizationApp()
{
    destroyCudaResources();
    destroyOpenGLResources();
    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

void VisualizationApp::run()
{
    initWindow();
    initOpenGLResources();
    initCudaResources();

    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window_))
    {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta = now - last;
        last = now;

        update(delta.count());
        draw();

        glfwPollEvents();
    }
}

void VisualizationApp::initWindow()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwSetErrorCallback(glfwErrorCallback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!window_)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    const GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        throw std::runtime_error("Failed to initialize GLEW: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(err))));
    }

    // GLEW may set a benign GL_INVALID_ENUM when initializing core profiles; clear it.
    glGetError();

    glViewport(0, 0, width_, height_);
    glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
    glEnable(GL_DEPTH_TEST);
}

void VisualizationApp::initOpenGLResources()
{
    compileShaders();

    // Full-screen quad for the heatmap texture.
    const float quadVertices[] = {
        // positions        // texcoords
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
         0.0f, -1.0f, 0.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
         0.0f,  1.0f, 0.0f,  1.0f, 1.0f,
    };

    glGenVertexArrays(1, &quadVao_);
    glBindVertexArray(quadVao_);
    glGenBuffers(1, &quadVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

    glBindVertexArray(0);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenVertexArrays(1, &pointVao_);
    glBindVertexArray(pointVao_);
    glGenBuffers(1, &pointVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, pointVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(kPointCount) * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), reinterpret_cast<void*>(0));
    glBindVertexArray(0);

    checkOpenGlError("initializing OpenGL resources");
}

void VisualizationApp::initCudaResources()
{
    throwIfCudaFailed(cudaGraphicsGLRegisterBuffer(&cudaPboResource_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard),
                      "registering heatmap pixel buffer with CUDA");
    throwIfCudaFailed(cudaGraphicsGLRegisterBuffer(&cudaVboResource_, pointVbo_, cudaGraphicsRegisterFlagsWriteDiscard),
                      "registering point VBO with CUDA");
}

void VisualizationApp::destroyCudaResources()
{
    if (cudaPboResource_ != nullptr)
    {
        const cudaError_t err = cudaGraphicsUnregisterResource(cudaPboResource_);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to unregister CUDA PBO resource: " << cudaGetErrorString(err) << '\n';
        }
        cudaPboResource_ = nullptr;
    }
    if (cudaVboResource_ != nullptr)
    {
        const cudaError_t err = cudaGraphicsUnregisterResource(cudaVboResource_);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to unregister CUDA VBO resource: " << cudaGetErrorString(err) << '\n';
        }
        cudaVboResource_ = nullptr;
    }
}

void VisualizationApp::destroyOpenGLResources()
{
    if (window_ != nullptr)
    {
        glfwMakeContextCurrent(window_);
    }
    if (pointVbo_ != 0)
    {
        glDeleteBuffers(1, &pointVbo_);
        pointVbo_ = 0;
    }
    if (pointVao_ != 0)
    {
        glDeleteVertexArrays(1, &pointVao_);
        pointVao_ = 0;
    }
    if (pbo_ != 0)
    {
        glDeleteBuffers(1, &pbo_);
        pbo_ = 0;
    }
    if (texture_ != 0)
    {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    if (quadVbo_ != 0)
    {
        glDeleteBuffers(1, &quadVbo_);
        quadVbo_ = 0;
    }
    if (quadVao_ != 0)
    {
        glDeleteVertexArrays(1, &quadVao_);
        quadVao_ = 0;
    }
    if (heatmapProgram_ != 0)
    {
        glDeleteProgram(heatmapProgram_);
        heatmapProgram_ = 0;
    }
    if (pointProgram_ != 0)
    {
        glDeleteProgram(pointProgram_);
        pointProgram_ = 0;
    }
}

void VisualizationApp::update(float deltaSeconds)
{
    elapsedTime_ += deltaSeconds;

    if (cudaPboResource_ != nullptr)
    {
        throwIfCudaFailed(cudaGraphicsMapResources(1, &cudaPboResource_, 0), "map heatmap buffer");
        uchar4* surface = nullptr;
        size_t size = 0;
        throwIfCudaFailed(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&surface), &size, cudaPboResource_),
                          "get mapped pointer for heatmap buffer");
        const size_t expected = static_cast<size_t>(width_) * static_cast<size_t>(height_) * sizeof(uchar4);
        if (size < expected)
        {
            throw std::runtime_error("Heatmap buffer size is smaller than expected");
        }
        cuda::fillHeatmap(surface, width_, height_, elapsedTime_);
        throwIfCudaFailed(cudaGraphicsUnmapResources(1, &cudaPboResource_, 0), "unmap heatmap buffer");
    }

    if (cudaVboResource_ != nullptr)
    {
        throwIfCudaFailed(cudaGraphicsMapResources(1, &cudaVboResource_, 0), "map point buffer");
        float3* points = nullptr;
        size_t size = 0;
        throwIfCudaFailed(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&points), &size, cudaVboResource_),
                          "get mapped pointer for point buffer");
        const size_t expected = static_cast<size_t>(kPointCount) * sizeof(float3);
        if (size < expected)
        {
            throw std::runtime_error("Point buffer size is smaller than expected");
        }
        cuda::fillPointCloud(points, kPointCount, elapsedTime_);
        throwIfCudaFailed(cudaGraphicsUnmapResources(1, &cudaVboResource_, 0), "unmap point buffer");
    }
}

void VisualizationApp::draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawHeatmap();
    drawPointCloud();

    glfwSwapBuffers(window_);
}

void VisualizationApp::drawHeatmap()
{
    const int halfWidth = width_ / 2;
    glViewport(0, 0, halfWidth, height_);

    glDisable(GL_DEPTH_TEST);
    glUseProgram(heatmapProgram_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glUniform1i(glGetUniformLocation(heatmapProgram_, "uTexture"), 0);

    glBindVertexArray(quadVao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_DEPTH_TEST);
}

void VisualizationApp::drawPointCloud()
{
    const int halfWidth = width_ / 2;
    glViewport(halfWidth, 0, width_ - halfWidth, height_);

    glUseProgram(pointProgram_);

    const float aspect = static_cast<float>(width_ - halfWidth) / static_cast<float>(height_);
    Mat4 projection = makePerspective(45.0f * kPi / 180.0f, aspect, 0.1f, 10.0f);
    const float orbitRadius = 3.5f;
    const float camX = std::cos(elapsedTime_ * 0.35f) * orbitRadius;
    const float camZ = std::sin(elapsedTime_ * 0.35f) * orbitRadius;
    Mat4 view = makeLookAt(camX, 2.0f, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    Mat4 mvp = multiply(projection, view);

    const int mvpLocation = glGetUniformLocation(pointProgram_, "uMVP");
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, mvp.data());
    glUniform1f(glGetUniformLocation(pointProgram_, "uTime"), elapsedTime_);

    glBindVertexArray(pointVao_);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, kPointCount);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(0);
}

void VisualizationApp::compileShaders()
{
    const char* heatmapVertexSrc = R"( #version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main()
{
    vTexCoord = aTexCoord;
    gl_Position = vec4(aPos, 1.0);
}
)";

    const char* heatmapFragmentSrc = R"( #version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    FragColor = texture(uTexture, vTexCoord);
}
)";

    const char* pointVertexSrc = R"( #version 430 core
layout(location = 0) in vec3 aPosition;

uniform mat4 uMVP;
uniform float uTime;

out float vDepth;
out float vTime;

void main()
{
    vec4 worldPosition = vec4(aPosition, 1.0);
    gl_Position = uMVP * worldPosition;
    gl_PointSize = 2.0 + 6.0 * abs(sin(uTime + aPosition.y * 2.5));
    vDepth = gl_Position.z;
    vTime = uTime;
}
)";

    const char* pointFragmentSrc = R"( #version 430 core
in float vDepth;
in float vTime;
out vec4 FragColor;

void main()
{
    float glow = exp(-abs(vDepth)) * 0.8 + 0.2;
    float pulse = 0.6 + 0.4 * sin(vTime * 1.5 + gl_FragCoord.x * 0.01);
    vec3 base = vec3(0.1, 0.6, 1.0);
    vec3 color = mix(base, vec3(1.0, 0.3, 0.8), pulse);
    FragColor = vec4(color * glow, 1.0);
}
)";

    unsigned int heatmapVert = compileShader(GL_VERTEX_SHADER, heatmapVertexSrc);
    unsigned int heatmapFrag = compileShader(GL_FRAGMENT_SHADER, heatmapFragmentSrc);
    heatmapProgram_ = linkProgram(heatmapVert, heatmapFrag);
    glDeleteShader(heatmapVert);
    glDeleteShader(heatmapFrag);

    unsigned int pointVert = compileShader(GL_VERTEX_SHADER, pointVertexSrc);
    unsigned int pointFrag = compileShader(GL_FRAGMENT_SHADER, pointFragmentSrc);
    pointProgram_ = linkProgram(pointVert, pointFrag);
    glDeleteShader(pointVert);
    glDeleteShader(pointFrag);
}

unsigned int VisualizationApp::compileShader(unsigned int type, const char* source)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        int logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> infoLog(static_cast<size_t>(logLength));
        glGetShaderInfoLog(shader, logLength, nullptr, infoLog.data());
        std::string log(infoLog.begin(), infoLog.end());
        throw std::runtime_error("Failed to compile shader: " + log);
    }
    return shader;
}

unsigned int VisualizationApp::linkProgram(unsigned int vertexShader, unsigned int fragmentShader)
{
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        int logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> infoLog(static_cast<size_t>(logLength));
        glGetProgramInfoLog(program, logLength, nullptr, infoLog.data());
        std::string log(infoLog.begin(), infoLog.end());
        throw std::runtime_error("Failed to link shader program: " + log);
    }
    return program;
}

} // namespace viz
