#include "utils.h"
// 全局变量


void init_spdlog_once()
{
    static std::once_flag flag; // 静态存储，整个程序生命周期内保持
    std::call_once(flag, []() {
        try
        {
            // 创建控制台（终端）接收器
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info); // 设置控制台日志等级
            // 设置日志格式. 参数含义: [日期][日志级别] [数据]
            // 补充 [线程号%t][文件名%s 函数名%#:行号%!]
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S][%t][%l] %v"); // 设置日志格式

            // 创建文件接收器
            // auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/DailyLog.log", true);
            auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/DailyLog.log", 0, 0);
            file_sink->set_level(spdlog::level::trace);               // 设置文件日志等级
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S][%t][%l] %v"); // 设置日志格式

            // auto qt_sink = std::make_shared<spdlog::sinks::qt_sink_mt>(ui->textEdit_log, "append");
            // qt_sink->set_level(spdlog::level::debug); // 设置文件日志等级
            // qt_sink->set_pattern("%l[%H:%M:%S] %v"); // 设置日志格式

            // 创建一个多重接收器的 logger
            // std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink, qt_sink};
            std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
            auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());

            // 设置全局 logger
            spdlog::set_default_logger(logger);
            // 设置全局日志等级为 debug，所有日志都会输出
            spdlog::set_level(spdlog::level::debug);
            // 开启日志刷新
            spdlog::flush_on(spdlog::level::info);
            // 示例日志
            spdlog::info("log class has been initialed !");
            // 手动刷新日志，确保写入到文件
            // spdlog::flush_every(std::chrono::seconds(1));  // 每秒刷新日志

        } catch (const spdlog::spdlog_ex &ex)
        {
            std::cout << "日志初始化失败: " << ex.what() << std::endl;
        }
    });
}

HalconCpp::HObject UTILS::__Mat1ToHObject__(const cv::Mat& image)
{
    HalconCpp::HObject hObj = HalconCpp::HObject();
    GenImage1(&hObj, "byte", image.cols, image.rows, (Hlong)image.ptr(0));
    return hObj;
}

HalconCpp::HObject UTILS::MatToHObject(const cv::Mat& image)
{
    HalconCpp::HObject hObj;
    if (image.type() == CV_8UC1)
        return __Mat1ToHObject__(image);
    else if (image.type() == CV_8UC3)
    {
        std::vector<cv::Mat> vec;
        cv::split(image, vec);
        cv::Mat imgB = vec[0];
        cv::Mat imgG = vec[1];
        cv::Mat imgR = vec[2];
        HalconCpp::HObject himgR = __Mat1ToHObject__(imgR);
        HalconCpp::HObject himgG = __Mat1ToHObject__(imgG);
        HalconCpp::HObject himgB = __Mat1ToHObject__(imgB);
        HalconCpp::Compose3(himgR, himgG, himgB, &hObj);
    }
    return hObj;
}

cv::Mat UTILS::HImageToMat(const HalconCpp::HImage& hImg)
{
    cv::Mat mat;
    int channels = hImg.CountChannels()[0].I();
    HalconCpp::HImage hImage = hImg.ConvertImageType("byte");
    Hlong hW = 0, hH = 0;
    HString cType;
    if (channels == 1)
    {
        void* r = hImage.GetImagePointer1(&cType, &hW, &hH);
        mat.create(int(hH), int(hW), CV_8UC1);
        memcpy(mat.data, static_cast<unsigned char*>(r), int(hW * hH));
    }
    else if (channels == 3)
    {
        void* r = NULL, * g = NULL, * b = NULL;
        hImage.GetImagePointer3(&r, &g, &b, &cType, &hW, &hH);
        mat.create(int(hH), int(hW), CV_8UC3);
        std::vector<cv::Mat> vec(3);
        vec[0].create(int(hH), int(hW), CV_8UC1);
        vec[1].create(int(hH), int(hW), CV_8UC1);
        vec[2].create(int(hH), int(hW), CV_8UC1);
        memcpy(vec[2].data, static_cast<unsigned char*>(r), int(hW * hH));
        memcpy(vec[1].data, static_cast<unsigned char*>(g), int(hW * hH));
        memcpy(vec[0].data, static_cast<unsigned char*>(b), int(hW * hH));
        cv::merge(vec, mat);
    }
    return mat;
}
