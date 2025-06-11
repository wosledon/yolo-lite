#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

// 检测结果结构体，包含矩形框、类别标签和置信度
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// 图像预处理：缩放到目标尺寸并转换为ncnn的Mat格式
void preprocess(const cv::Mat &img, ncnn::Mat &in, int target_size)
{
    cv::Mat resized;
    // 缩放图像到 target_size x target_size
    cv::resize(img, resized, cv::Size(target_size, target_size));
    // 转换为ncnn输入格式（BGR）
    in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, target_size, target_size);
}

// 后处理：解析模型输出，筛选目标并进行NMS
void postprocess(const ncnn::Mat &out, std::vector<Object> &objects, float prob_threshold, float nms_threshold)
{
    // 这里需要根据你的模型输出格式进行解析
    // 通常YOLOv3输出为[N, 85]，N为检测框数，85为4+1+80
    // 这里只给出伪代码，具体实现需参考你的模型输出
    // for (int i = 0; i < out.h; i++) {
    //     float* values = out.row(i);
    //     float score = values[4];
    //     if (score > prob_threshold) {
    //         // ...解析类别和坐标...
    //         objects.push_back(...);
    //     }
    // }
    // NMS处理
    // ...existing code...
}

// 目标检测主流程：输入图像，输出检测结果
void detect_objects(ncnn::Net &net, const cv::Mat &img, std::vector<Object> &objects, int target_size = 416, float prob_threshold = 0.5f, float nms_threshold = 0.45f)
{
    ncnn::Mat in;
    // 图像预处理
    preprocess(img, in, target_size);

    // 创建ncnn推理器
    ncnn::Extractor ex = net.create_extractor();
    // 设置输入
    ex.input("data", in);

    ncnn::Mat out;
    // 获取输出
    ex.extract("output", out);

    // 后处理，得到检测结果
    postprocess(out, objects, prob_threshold, nms_threshold);
}

int main(int argc, char **argv)
{
    // 支持三种用法：
    // 1. ./yolo3_ncnn model.param model.bin           // 从摄像头读取
    // 2. ./yolo3_ncnn model.param model.bin image.jpg // 从图片读取
    if (argc != 3 && argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " model.param model.bin [image.jpg]" << std::endl;
        return -1;
    }

    const char *param_path = argv[1];
    const char *bin_path = argv[2];

    // 加载ncnn模型
    ncnn::Net net;
    net.load_param(param_path);
    net.load_model(bin_path);

    cv::Mat frame;
    if (argc == 4)
    {
        // 从图片读取
        frame = cv::imread(argv[3]);
        if (frame.empty())
        {
            std::cerr << "Image open failed!" << std::endl;
            return -1;
        }
    }
    else
    {
        // 从摄像头读取
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            std::cerr << "Camera open failed!" << std::endl;
            return -1;
        }
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Capture failed!" << std::endl;
            return -1;
        }
    }

    // 检测目标
    std::vector<Object> objects;
    detect_objects(net, frame, objects);

    // 绘制检测结果并输出
    for (const auto &obj : objects)
    {
        cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
        std::cout << "label: " << obj.label << " prob: " << obj.prob << std::endl;
    }
    // 保存结果图片
    cv::imwrite("result.jpg", frame);

    return 0;
}