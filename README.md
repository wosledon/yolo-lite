# yolo-lite

## 简介

本项目基于 ncnn 和 OpenCV 实现 YOLOv4 目标检测的简易推理与演示。

## 依赖

- OpenCV 4.x
- ncnn

### 安装依赖

#### Windows 推荐方式

1. **OpenCV**  
   - 推荐用 [vcpkg](https://github.com/microsoft/vcpkg)：  
     ```sh
     git clone https://github.com/microsoft/vcpkg.git
     cd vcpkg
     .\bootstrap-vcpkg.bat
     .\vcpkg install opencv4
     ```
     安装完成后，设置 `OpenCV_DIR` 为 vcpkg 安装路径下的 `installed\x64-windows\share\opencv4`。
   - 或手动下载并解压 [OpenCV 官方预编译包](https://opencv.org/releases/)，并设置 `OpenCV_DIR` 为解压路径下的 `build` 目录。

2. **ncnn**  
   - 推荐参考 [ncnn官方文档](https://github.com/Tencent/ncnn/wiki/how-to-build) 使用 CMake 编译：
     ```sh
     git clone https://github.com/Tencent/ncnn.git
     cd ncnn
     mkdir build && cd build
     cmake .. -DCMAKE_BUILD_TYPE=Release
     cmake --build . --config Release
     ```
     编译完成后，设置 `ncnn_DIR` 为 `ncnn/build/install/lib/cmake/ncnn`。

#### Linux/macOS

- **OpenCV**  
  - Ubuntu/Debian:
    ```sh
    sudo apt update
    sudo apt install libopencv-dev
    ```
  - macOS (Homebrew):
    ```sh
    brew install opencv
    ```
  - 或源码编译，参考 [OpenCV 官方文档](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)。

- **ncnn**  
  - 推荐源码编译：
    ```sh
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install
    ```
    安装后，`ncnn_DIR` 通常为 `/usr/local/lib/cmake/ncnn`。

## 编译

```sh
mkdir build
cd build
cmake ..
cmake --build .
```

如使用 vcpkg 安装依赖，推荐如下方式编译（假设 vcpkg 路径为 D:/tools/vcpkg）：
```sh
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=D:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

如未自动找到OpenCV或ncnn，可在 `CMakeLists.txt` 中手动设置 `OpenCV_DIR` 和 `ncnn_DIR`，例如：
```cmake
set(OpenCV_DIR "your/opencv/path")
set(ncnn_DIR "your/ncnn/path")
```

## 运行

```sh
# 从摄像头读取
./yolo4_ncnn model.param model.bin

# 或从图片读取
./yolo4_ncnn model.param model.bin image.jpg
```

## 其他

- 检测结果会保存为 `result.jpg`。
- 代码结构和用法详见 `yolo4_ncnn.cpp` 注释。

## 基础模型地址

https://github.com/nihui/ncnn-assets/tree/master

## demo

```sh
PS D:\repos\github\yolo-lite> cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=D:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake

PS D:\repos\github\yolo-lite> cmake --build build

PS D:\repos\github\yolo-lite\build\Debug> .\yolo4_ncnn.exe ..\..\model\yolov4-tiny-opt.param ..\..\model\yolov4-tiny-opt.bin ..\..\model\person.jpg
```