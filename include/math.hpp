#pragma once

#include <array>

namespace viz {

using Mat4 = std::array<float, 16>;

Mat4 makeIdentity();
Mat4 makePerspective(float fovyRadians, float aspect, float nearPlane, float farPlane);
Mat4 makeLookAt(float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ,
                float upX, float upY, float upZ);
Mat4 multiply(const Mat4& lhs, const Mat4& rhs);

} // namespace viz
