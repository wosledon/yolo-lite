cmake_minimum_required(VERSION 3.10)
project(yolo3_ncnn_demo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(ncnn REQUIRED)

add_executable(yolo3_ncnn yolo3_ncnn.cpp)

target_link_libraries(yolo3_ncnn PRIVATE ${OpenCV_LIBS} ncnn)
