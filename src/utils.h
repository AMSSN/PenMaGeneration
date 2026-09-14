#ifndef UTILS_H
#define UTILS_H
#include <algorithm> // 必须包含此头文件以使用 std::sort
#include <iostream>
#include <cassert>
#include <codecvt>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <filesystem>
// log
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> // For colored console logs
#include <spdlog/sinks/basic_file_sink.h>    // For file logging
#include <spdlog/sinks/daily_file_sink.h>
// #include <spdlog/sinks/qt_sinks.h>           // For QT widget logging

// 3rdpartys
#include "halconcpp/HalconCpp.h"
#include "opencv.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include "halconcpp/HalconCpp.h"
using namespace HalconCpp;
namespace fs = std::filesystem;



namespace UTILS
{
	//void UTILS::loadConfigGlobal();
	//void UTILS::saveConfigGlobal();
	HalconCpp::HObject __Mat1ToHObject__(const cv::Mat& image);
	HalconCpp::HObject MatToHObject(const cv::Mat& image);
	cv::Mat HImageToMat(const HalconCpp::HImage& hImg);



} // namespace UTILS
void init_spdlog_once();

#endif // UTILS_H
