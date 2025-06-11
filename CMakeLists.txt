cmake_minimum_required(VERSION 3.10)
project(yolo4_ncnn_demo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(yolo4_ncnn yolo4_ncnn.cpp)

target_link_libraries(yolo4_ncnn PRIVATE ${OpenCV_LIBS} ncnn)
