cmake_minimum_required(VERSION 3.22)
project(cuda_visualization_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(cuda_kernels STATIC
    src/cuda_kernels.cu
)

target_include_directories(cuda_kernels
    PUBLIC
        include
)

target_compile_options(cuda_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(cuda_visualization
    src/main.cpp
    src/visualization_app.cu
    src/math.cpp
)

target_include_directories(cuda_visualization
    PRIVATE
        include
)

set_target_properties(cuda_visualization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(cuda_visualization
    PRIVATE
        cuda_kernels
        OpenGL::GL
        glfw
        GLEW::GLEW
        CUDA::cudart
        CUDA::cuda_driver
)

if (MSVC)
    target_compile_options(cuda_visualization PRIVATE /W4 /permissive-)
    target_compile_options(cuda_kernels PRIVATE /W4 /permissive-)
else()
    target_compile_options(cuda_visualization PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(cuda_kernels PRIVATE -Wall -Wextra -Wpedantic)
endif()
