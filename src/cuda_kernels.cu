#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace viz {
namespace cuda {
namespace {

constexpr float kPi = 3.14159265358979323846f;

inline void throwIfCudaFailed(cudaError_t error, const char* action)
{
    if (error != cudaSuccess)
    {
        std::ostringstream oss;
        oss << action << ": " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

__global__ void heatmapKernel(uchar4* surface, int width, int height, float timeSeconds)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const float fx = (static_cast<float>(x) / static_cast<float>(width)) * 2.0f - 1.0f;
    const float fy = (static_cast<float>(y) / static_cast<float>(height)) * 2.0f - 1.0f;

    const float radius = sqrtf(fx * fx + fy * fy);
    const float angle = atan2f(fy, fx);

    const float wave1 = __sinf(6.0f * radius - timeSeconds * 2.0f);
    const float wave2 = __cosf(4.0f * angle + timeSeconds * 1.5f);
    const float wave3 = __sinf(10.0f * (fx + fy) + timeSeconds * 0.5f);

    float intensity = 0.5f + 0.5f * (0.5f * wave1 + 0.3f * wave2 + 0.2f * wave3);
    intensity = fminf(fmaxf(intensity, 0.0f), 1.0f);

    const float hue = fmodf(timeSeconds * 0.1f + radius * 0.35f + angle * 0.1f, 1.0f);
    const float saturation = 0.8f;
    const float value = intensity;

    const float c = value * saturation;
    const float xComp = c * (1.0f - fabsf(fmodf(hue * 6.0f, 2.0f) - 1.0f));
    const float m = value - c;

    float r, g, b;
    if (hue < 1.0f / 6.0f)
    {
        r = c; g = xComp; b = 0.0f;
    }
    else if (hue < 2.0f / 6.0f)
    {
        r = xComp; g = c; b = 0.0f;
    }
    else if (hue < 3.0f / 6.0f)
    {
        r = 0.0f; g = c; b = xComp;
    }
    else if (hue < 4.0f / 6.0f)
    {
        r = 0.0f; g = xComp; b = c;
    }
    else if (hue < 5.0f / 6.0f)
    {
        r = xComp; g = 0.0f; b = c;
    }
    else
    {
        r = c; g = 0.0f; b = xComp;
    }

    const unsigned char red = static_cast<unsigned char>((r + m) * 255.0f);
    const unsigned char green = static_cast<unsigned char>((g + m) * 255.0f);
    const unsigned char blue = static_cast<unsigned char>((b + m) * 255.0f);

    surface[y * width + x] = make_uchar4(red, green, blue, 255);
}

__global__ void pointCloudKernel(float3* points, int count, float timeSeconds)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }

    const float normalized = static_cast<float>(idx) / static_cast<float>(count);
    const float revolutions = 12.0f;
    const float angle = normalized * revolutions * 2.0f * kPi + timeSeconds * 0.3f;
    const float spiral = normalized * 2.0f - 1.0f;

    const float radius = 0.2f + 0.6f * normalized + 0.1f * __sinf(timeSeconds + normalized * 10.0f);
    const float y = spiral + 0.2f * __cosf(angle * 1.5f + timeSeconds);

    const float x = radius * __cosf(angle);
    const float z = radius * __sinf(angle);

    points[idx] = make_float3(x, y, z);
}

} // namespace

void fillHeatmap(uchar4* surface, int width, int height, float timeSeconds)
{
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    heatmapKernel<<<grid, block>>>(surface, width, height, timeSeconds);
    throwIfCudaFailed(cudaPeekAtLastError(), "launch heatmap kernel");
    throwIfCudaFailed(cudaDeviceSynchronize(), "synchronize heatmap kernel");
}

void fillPointCloud(float3* points, int count, float timeSeconds)
{
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    pointCloudKernel<<<blocks, threads>>>(points, count, timeSeconds);
    throwIfCudaFailed(cudaPeekAtLastError(), "launch point cloud kernel");
    throwIfCudaFailed(cudaDeviceSynchronize(), "synchronize point cloud kernel");
}

} // namespace cuda
} // namespace viz
