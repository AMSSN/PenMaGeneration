#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "HalconCpp.h"
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <DotNumber.h>
#include <penma_batch.h>


namespace fs = std::filesystem;
using namespace HalconCpp;




// 批量生成示例:
//   PenMaGeneration.exe batch <txt> <背景文件夹> [输出根目录]
//   默认: txt="samples.txt"  背景="bgs"  输出="yoloset"
int main() {
    const std::string txt = "samples.txt";
    const std::string bgd = "bg";
    const std::string out = "yoloset";

    penma_batch::BatchConfig cfg;
    cfg.dotSize   = 13;
    cfg.dotSpac   = 2;
    cfg.numSpac   = 25;
    cfg.lineSpac  = 30;
    cfg.maskRotate = 2.0;
    cfg.driftPix  = 1;
    cfg.driftConf = 0.9;
    cfg.filterK   = 3;
    cfg.genYolo   = true;   // 功能2: 同步输出 yolo 标签
    cfg.boxPad    = 2;      // 每个数字框向外扩充 2 像素

    const int n = penma_batch::batch_generate(txt, bgd, out, cfg);
    if (n < 0)
    {
        std::cout << "batch 生成失败, code=" << n
                    << " (-1=txt 读取失败, -2=背景目录无可用图片)\n";
        std::cout << "用法: 程序 batch <txt> <背景文件夹> [输出根目录]\n";
    }
    else
    {
        std::cout << "batch 生成完成, 共 " << n << " 张\n";
        std::cout << "  图片 -> " << out << "/images\n"
                    << "  标签 -> " << out << "/labels\n";
    }
    return 0;
    

    //std::cout << "hello world!\n";

    //DotNumber dn;
    //std::vector<std::string> nums = { "26040","20260","2028" };
    //dn.set_numbers(nums);
    //dn.dotSize = 13;
    //dn.dotSpac = 2;
    //dn.numSpac = 25;
    //dn.lineSpac = 30;
    //dn.maskRotate = 2; // 整体旋转正负maskRotate度

    //dn.gen_mask_mat();          // 生成基础掩码
    //cv::Mat base_mask = dn.get_mask_result(false);  // 白底黑字
    //cv::imwrite("./base_mask.png", base_mask);

    //dn.add_style_drift(1, 0.9);// 墨点位置漂移
    //dn.add_style_filter(3);     // 模糊，得到软边 alpha
    ////dn.add_style_defect(0.05, 0.03, 0.10);  // 缺墨 / 飞墨 / 拖尾
    //cv::Mat mask = dn.get_mask_result(false);  // 白底黑字
    //cv::Mat alpha = dn.get_mask_result(true);   // 墨点=255
    //cv::imwrite("./mask.png", mask);
    //cv::imwrite("./alpha.png", alpha);

    //cv::Mat bg = cv::imread("kongbai_rotate.jpg", cv::IMREAD_COLOR);
    ////cv::Mat out = dn.paste_mask(bg, cv::Point(150, 150));
    //cv::Mat out = dn.paste_mask(bg, cv::Point(35, 300), cv::Scalar(0, 0, 0));
    //cv::imwrite("./out.png", out);
    //return 0;
}


