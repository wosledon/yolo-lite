cmake_minimum_required(VERSION 3.10)
project(yolo3_ncnn_demo)

set(CMAKE_CXX_STANDARD 11)

# 查找OpenCV（如未自动找到，可手动指定OpenCV_DIR）
set(OpenCV_DIR "D:/tools/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

# 查找ncnn（如未自动找到，可手动指定ncnn_DIR）
# set(ncnn_DIR "your/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_executable(yolo3_ncnn yolo3_ncnn.cpp)

target_link_libraries(yolo3_ncnn PRIVATE ${OpenCV_LIBS} ncnn)
