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
    // 调试输出 shape
    std::cout << "out dims: " << out.dims << ", w: " << out.w << ", h: " << out.h << ", c: " << out.c << std::endl;
    // 可选：打印前几行内容
    for (int i = 0; i < std::min(5, out.h); i++)
    {
        const float *values = out.row(i);
        for (int j = 0; j < std::min(10, out.w); j++)
        {
            std::cout << values[j] << " ";
        }
        std::cout << std::endl;
    }
    // YOLOv4输出为[N, 85]，N为检测框数，85=4(bbox)+1(obj)+80(class)
    const int num_classes = 80;
    const int num_fields = 5 + num_classes; // 4+1+80

    std::vector<Object> proposals;

    for (int i = 0; i < out.h; i++)
    {
        const float *values = out.row(i);
        float obj_score = values[4];
        if (obj_score < prob_threshold)
            continue;

        // 找到最大类别分数及其索引
        float max_class_score = 0.f;
        int class_index = -1;
        for (int j = 0; j < num_classes; j++)
        {
            float class_score = values[5 + j];
            if (class_score > max_class_score)
            {
                max_class_score = class_score;
                class_index = j;
            }
        }

        float confidence = obj_score * max_class_score;
        if (confidence < prob_threshold)
            continue;

        // bbox: [center_x, center_y, w, h] -> [x0, y0, x1, y1]
        float cx = values[0];
        float cy = values[1];
        float w = values[2];
        float h = values[3];
        float x0 = cx - w * 0.5f;
        float y0 = cy - h * 0.5f;
        float x1 = cx + w * 0.5f;
        float y1 = cy + h * 0.5f;

        Object obj;
        obj.rect = cv::Rect_<float>(x0, y0, w, h);
        obj.label = class_index;
        obj.prob = confidence;
        proposals.push_back(obj);
    }

    // NMS
    std::sort(proposals.begin(), proposals.end(), [](const Object &a, const Object &b)
              { return a.prob > b.prob; });

    std::vector<int> picked;
    for (size_t i = 0; i < proposals.size(); i++)
    {
        const Object &a = proposals[i];
        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Object &b = proposals[picked[j]];
            // 计算IoU
            float inter_area = (a.rect & b.rect).area();
            float union_area = a.rect.area() + b.rect.area() - inter_area;
            if (inter_area / union_area > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }

    objects.clear();
    for (size_t i = 0; i < picked.size(); i++)
        objects.push_back(proposals[picked[i]]);
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
    // 1. ./yolo4_ncnn model/yolov4-tiny-opt.param model/yolov4-tiny-opt.bin           // 从摄像头读取
    // 2. ./yolo4_ncnn model/yolov4-tiny-opt.param model/yolov4-tiny-opt.bin image.jpg // 从图片读取
    // 模型文件应放在 model 文件夹下
    if (argc != 3 && argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " model/yolov4-tiny-opt.param model/yolov4-tiny-opt.bin [image.jpg]" << std::endl;
        return -1;
    }

    const char *param_path = argv[1];
    const char *bin_path = argv[2];

    // 加载ncnn模型
    ncnn::Net net;
    if (net.load_param(param_path) != 0 || net.load_model(bin_path) != 0)
    {
        std::cerr << "Failed to load ncnn model!" << std::endl;
        return -1;
    }

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
    if (!cv::imwrite("result.jpg", frame))
    {
        std::cerr << "Failed to save result.jpg" << std::endl;
        return -1;
    }

    std::cout << "Result saved to result.jpg" << std::endl;
    return 0;
}