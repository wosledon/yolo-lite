# yolo-lite

## 简介

本项目基于 ncnn 和 OpenCV 实现 YOLOv3 目标检测的简易推理与演示。

## 依赖

- OpenCV 4.x
- ncnn

### 安装依赖

#### Windows 推荐方式

1. **OpenCV**  
   - 推荐用 [vcpkg](https://github.com/microsoft/vcpkg)：  
     ```
     vcpkg install opencv4
     ```
   - 或手动下载并解压官方预编译包。

2. **ncnn**  
   - 参考 [ncnn官方文档](https://github.com/Tencent/ncnn/wiki/how-to-build) 编译或下载预编译包。

#### Linux/macOS

- 推荐用包管理器或源码编译安装 OpenCV 和 ncnn。

## 编译

```sh
mkdir build
cd build
cmake ..
cmake --build .
```

如未自动找到OpenCV或ncnn，可在 `CMakeLists.txt` 中手动设置 `OpenCV_DIR` 和 `ncnn_DIR`。

## 运行

```sh
# 从摄像头读取
./yolo3_ncnn model.param model.bin

# 或从图片读取
./yolo3_ncnn model.param model.bin image.jpg
```

## 其他

- 检测结果会保存为 `result.jpg`。
- 代码结构和用法详见 `yolo3_ncnn.cpp` 注释。

## 基础模型地址

https://github.com/nihui/ncnn-assets/tree/master