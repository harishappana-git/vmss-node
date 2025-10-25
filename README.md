# CUDA Real-Time Visualisation Tools

This project provides a starting point for visualising data produced by CUDA kernels in real time. It contains a small host application written in modern C++ that renders a 2D heatmap and a 3D point-cloud simultaneously. Both visualisations are updated every frame by CUDA kernels via CUDA/OpenGL interoperability, which keeps the data entirely on the GPU.

## Features

- **GPU-only data path** – pixel buffers and vertex buffers are shared between CUDA and OpenGL so no CPU copies are required.
- **Real-time 2D heatmap** – demonstrates how to generate a colour-mapped scalar field on the GPU and present it as a texture.
- **Animated 3D point cloud** – shows how to populate large vertex buffers in CUDA and render them as point sprites.
- **Modular kernels** – CUDA kernels live in `src/cuda_kernels.cu` and can be extended or replaced with application-specific logic.
- **CMake build** – cross-platform project setup with configurable dependencies.

## Repository layout

```
.
├── CMakeLists.txt          # Build configuration for the application and kernels
├── include/
│   ├── cuda_kernels.hpp    # Host-side declarations for CUDA kernels
│   ├── math.hpp            # Minimal linear-algebra helpers
│   └── visualization_app.hpp
├── src/
│   ├── cuda_kernels.cu     # Heatmap and point-cloud CUDA kernels
│   ├── main.cpp            # Application entry point
│   ├── math.cpp            # CPU implementations of math helpers
│   └── visualization_app.cu# GLFW/GLEW based renderer with CUDA interop
└── README.md
```

## Prerequisites

1. **GPU & Drivers**
   - NVIDIA GPU with Compute Capability 6.0+ recommended.
   - Latest NVIDIA drivers compatible with your CUDA toolkit.
2. **CUDA Toolkit 11.4+**
   - Required for compiling the kernels and CUDA/OpenGL interop headers.
3. **Build toolchain**
   - CMake 3.22 or newer.
   - A C++20-capable compiler (MSVC 2022, clang 14+, or GCC 11+).
4. **OpenGL dependencies**
   - GLFW (3.3 or newer) for window/context creation.
   - GLEW for OpenGL function loading.

### Installing dependencies

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install build-essential cmake libglfw3-dev libglew-dev
```
Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) via the official installer or package manager.

#### Fedora
```bash
sudo dnf install @development-tools cmake glfw-devel glew-devel
```
Install the CUDA toolkit using the RPM packages from NVIDIA.

#### Windows (Visual Studio)
- Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- Use [vcpkg](https://github.com/microsoft/vcpkg) to install dependencies:
  ```powershell
  vcpkg install glfw3 glew
  vcpkg integrate install
  ```
- Configure CMake with `-DCMAKE_TOOLCHAIN_FILE="<path-to-vcpkg>\scripts\buildsystems\vcpkg.cmake"`.

#### macOS
CUDA does not support current macOS releases. You can still study the host-side OpenGL scaffolding, but the CUDA kernels cannot be built or executed on macOS.

## Building

1. Clone the repository and create a build directory:
   ```bash
   git clone <your-repo-url>
   cd vmss-node
   cmake -S . -B build
   ```
2. Build the project:
   ```bash
   cmake --build build
   ```
   On Windows/MSVC you can pass `--config Release` for an optimised build.

If CMake cannot locate GLFW or GLEW automatically, set the corresponding hints during configuration, for example:
```bash
cmake -S . -B build \
  -Dglfw3_DIR=/path/to/glfw/lib/cmake/glfw3 \
  -DGLEW_ROOT=/path/to/glew
```

## Running the demo

After a successful build, launch the executable:
```bash
./build/cuda_visualization
```
This opens a window with two panes:

- **Left pane (2D heatmap)** – a GPU-generated scalar field that highlights how you can render CUDA-computed textures. Modify `viz::cuda::fillHeatmap` to visualise your own 2D data such as simulation slices or profiling metrics.
- **Right pane (3D point cloud)** – animated geometry that demonstrates updating large vertex buffers from CUDA. Adapt `viz::cuda::fillPointCloud` to map your own particle systems, fluid samples, or spatial instrumentation data.

Use this template by replacing the demo kernels with logic that interprets your CUDA program state. Because the buffers remain on the GPU, the render loop can scale to large datasets with minimal CPU overhead.

## Extending the toolset

- Add additional CUDA kernels in `src/cuda_kernels.cu` and declare them in `include/cuda_kernels.hpp` to stream more datasets into the visualiser.
- Create more rendering passes by following the patterns in `src/visualization_app.cu`—each pass can have its own shader program, buffer bindings, and viewport layout.
- Integrate ImGui or other UI libraries to expose runtime controls (colour maps, camera controls, kernel parameters).
- Hook into your existing C++/CUDA projects by including the `VisualizationApp` class and feeding it with simulation data in real time.

## Troubleshooting

- **Black window or no animation** – ensure that both CUDA and OpenGL are using the same GPU. On laptops with hybrid graphics, force the application to use the discrete NVIDIA GPU.
- **CUDA registration errors** – verify that the OpenGL context is current before registering buffers and that your driver supports CUDA/OpenGL interoperability.
- **Missing OpenGL extensions** – confirm that your GPU/driver supports OpenGL 4.3+. You can lower the requested version in `VisualizationApp::initWindow` if needed.

## License

This project is provided as-is without any specific licence. Adapt the code to your needs and integrate it into your workflows as required.
