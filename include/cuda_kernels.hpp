#pragma once

#include <cuda_runtime.h>

namespace viz {
namespace cuda {

void fillHeatmap(uchar4* surface, int width, int height, float timeSeconds);
void fillPointCloud(float3* points, int count, float timeSeconds);

} // namespace cuda
} // namespace viz
