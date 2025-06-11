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
void postprocess(const ncnn::Mat &out, std::vector<Object> &objects, float prob_threshold, float nms_threshold,
                 float scale_x, float scale_y, int pad_x, int pad_y, int orig_w, int orig_h)
{
    // 假设 out.w == 6: [x0, y0, x1, y1, score, label]
    std::vector<Object> proposals;
    for (int i = 0; i < out.h; i++)
    {
        const float *values = out.row(i);
        float score = values[4];
        if (score < prob_threshold || score > 1.5f)
            continue;
        Object obj;
        int target_size = 416;
        // 对于yolov4-tiny ncnn模型，输出通常是归一化的[x0, y0, x1, y1]，即左上和右下角，且都在[0,1]区间
        float x0i = values[0] * target_size;
        float y0i = values[1] * target_size;
        float x1i = values[2] * target_size;
        float y1i = values[3] * target_size;
        // 去padding和缩放
        float x0 = (x0i - pad_x) / scale_x;
        float y0 = (y0i - pad_y) / scale_y;
        float x1 = (x1i - pad_x) / scale_x;
        float y1 = (y1i - pad_y) / scale_y;
        if (x1 < x0)
            std::swap(x0, x1);
        if (y1 < y0)
            std::swap(y0, y1);
        x0 = std::max(std::min(x0, float(orig_w - 1)), 0.f);
        y0 = std::max(std::min(y0, float(orig_h - 1)), 0.f);
        x1 = std::max(std::min(x1, float(orig_w - 1)), 0.f);
        y1 = std::max(std::min(y1, float(orig_h - 1)), 0.f);

        obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);

        obj.label = static_cast<int>(values[5]);
        obj.prob = score;

        std::cout << "[DEBUG] yolo4-tiny: (" << values[0] << "," << values[1] << "," << values[2] << "," << values[3]
                  << ") mapped: (" << x0 << "," << y0 << "," << x1 << "," << y1 << ") rect: ("
                  << obj.rect.x << "," << obj.rect.y << "," << obj.rect.width << "," << obj.rect.height << ")"
                  << " label: " << obj.label << " prob: " << obj.prob << std::endl;

        proposals.push_back(obj);
    }

    // NMS
    objects.clear();
    std::vector<int> picked;
    for (size_t i = 0; i < proposals.size(); i++)
    {
        const Object &a = proposals[i];
        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Object &b = proposals[picked[j]];
            // 只对同类做NMS
            if (a.label != b.label)
                continue;
            float inter_area = (a.rect & b.rect).area();
            float union_area = a.rect.area() + b.rect.area() - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
    for (size_t i = 0; i < picked.size(); i++)
    {
        // 增加padding
        Object obj = proposals[picked[i]];
        float pad = 50.0f; // 每边增加5像素
        obj.rect.x = std::max(obj.rect.x - pad, 0.f);
        obj.rect.y = std::max(obj.rect.y - pad, 0.f);
        obj.rect.width = std::min(obj.rect.width + 2 * pad, float(orig_w) - obj.rect.x);
        obj.rect.height = std::min(obj.rect.height + 2 * pad, float(orig_h) - obj.rect.y);
        objects.push_back(obj);
    }
}

// 目标检测主流程：输入图像，输出检测结果
void detect_objects(ncnn::Net &net, const cv::Mat &img, std::vector<Object> &objects, int target_size = 416, float prob_threshold = 0.8f, float nms_threshold = 0.45f)
{
    // 等比例缩放+padding
    int w = img.cols;
    int h = img.rows;
    float scale = std::min(target_size / (w * 1.f), target_size / (h * 1.f));
    int new_w = int(w * scale);
    int new_h = int(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat input = cv::Mat::zeros(target_size, target_size, img.type());
    int dx = (target_size - new_w) / 2;
    int dy = (target_size - new_h) / 2;
    resized.copyTo(input(cv::Rect(dx, dy, new_w, new_h)));

    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR, target_size, target_size);

    // 记录缩放和padding参数
    float scale_x = scale;
    float scale_y = scale;
    int pad_x = dx;
    int pad_y = dy;

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    // 传递padding参数
    postprocess(out, objects, prob_threshold, nms_threshold, scale_x, scale_y, pad_x, pad_y, w, h);
}

// COCO 80类名称
const char *coco_names[] = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

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
        std::string label_text;
        if (obj.label >= 0 && obj.label < 80)
            label_text = std::string(coco_names[obj.label]);
        else
            label_text = "unknown";
        char prob_text[32];
        snprintf(prob_text, sizeof(prob_text), " %.2f", obj.prob);
        label_text += prob_text;

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = std::max(int(obj.rect.x), 0);
        int y = std::max(int(obj.rect.y), 0);
        cv::rectangle(frame, cv::Rect(x, y - label_size.height, label_size.width, label_size.height + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label_text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        std::cout << "label: " << obj.label << " (" << label_text << ")" << " prob: " << obj.prob << std::endl;
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