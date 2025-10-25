#include "math.hpp"

#include <cmath>

namespace viz {

Mat4 makeIdentity()
{
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    return m;
}

Mat4 makePerspective(float fovyRadians, float aspect, float nearPlane, float farPlane)
{
    const float f = 1.0f / std::tan(fovyRadians / 2.0f);
    Mat4 m{};
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    m[11] = -1.0f;
    m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    return m;
}

static std::array<float, 3> normalize(const std::array<float, 3>& v)
{
    const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len == 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    return {v[0] / len, v[1] / len, v[2] / len};
}

static std::array<float, 3> cross(const std::array<float, 3>& a, const std::array<float, 3>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

static float dot(const std::array<float, 3>& a, const std::array<float, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Mat4 makeLookAt(float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ,
                float upX, float upY, float upZ)
{
    const std::array<float, 3> eye{eyeX, eyeY, eyeZ};
    const std::array<float, 3> center{centerX, centerY, centerZ};
    const std::array<float, 3> up{upX, upY, upZ};

    auto f = normalize({center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]});
    auto upN = normalize(up);
    auto s = normalize(cross(f, upN));
    auto u = cross(s, f);

    Mat4 m = makeIdentity();
    m[0] = s[0];
    m[4] = s[1];
    m[8] = s[2];

    m[1] = u[0];
    m[5] = u[1];
    m[9] = u[2];

    m[2] = -f[0];
    m[6] = -f[1];
    m[10] = -f[2];

    m[12] = -dot(s, eye);
    m[13] = -dot(u, eye);
    m[14] = dot(f, eye);

    return m;
}

Mat4 multiply(const Mat4& lhs, const Mat4& rhs)
{
    Mat4 result{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            float value = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                value += lhs[row + i * 4] * rhs[i + col * 4];
            }
            result[row + col * 4] = value;
        }
    }
    return result;
}

} // namespace viz
