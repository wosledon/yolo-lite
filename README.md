# yolo-lite

## 一键安装依赖（推荐vcpkg）

1. 安装 [vcpkg](https://github.com/microsoft/vcpkg) 并配置环境变量。
2. 在项目根目录下执行：
   ```
   vcpkg install opencv4
   ```
3. 使用vcpkg集成CMake（推荐CMake 3.21+）：
   ```
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg根目录]/scripts/buildsystems/vcpkg.cmake
   cmake --build build
   ```

## 其他依赖

- ncnn（需自行编译或下载预编译包，并配置好include和lib路径）

## 运行方式

- 参考 `yolo3_ncnn.cpp` 注释说明。