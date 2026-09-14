#ifndef PENMA_BATCH_H
#define PENMA_BATCH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "DotNumber.h"

// ============================================================================
// 喷码合成 + YOLO 标注 批量生成
//
// 功能 1：
//   读取一个 txt 与一个背景文件夹，批量生成喷码融合图。
//   txt 格式：每行一个样本，行内由一个或多个数字串组成、用空格分隔，例如：
//       26040 20260 2028
//       26040
//       20260 2028
//   每个样本把行内数字串按“上下多行喷码”渲染，背景图循环使用
//   (每行文本对应一张背景，用完再从头取)，喷码整体随机落在背景左上 1/4 区域内融合。
//
// 功能 2：
//   可选同时输出 YOLO 格式标注：每个数字一个检测框，
//   框 = 数字最小外接矩形向外扩充 cfg.boxPad 像素(默认 2)，类别 = 数字本身 0~9。
//
// 统一保存结构：
//   <outRoot>/images/000000.png      (图片)
//   <outRoot>/labels/000000.txt      (yolo 标签, 每行: cls cx cy w h, 归一化)
//   文件名按生成顺序编号(默认 6 位，可配 cfg.nameWidth)。
// ============================================================================
namespace penma_batch
{

struct BatchConfig
{
    // —— 喷码点阵参数(对应 DotNumber 同名成员) ——
    int    dotSize   = 13;
    int    dotSpac   = 2;
    int    numSpac   = 25;     // 字符间距
    int    lineSpac  = 30;     // 行距(一行 = 一个数字串)
    int    sideSpac  = 2;
    int    sideSpacX = 50;
    double maskRotate = 2.0;   // 整体随机旋转范围 ±maskRotate(度)

    // —— 风格后处理 ——
    int    driftPix  = 1;      // 墨点漂移像素(<=0 关闭)
    double driftConf = 0.9;
    int    filterK   = 3;      // 模糊核大小(<=0 关闭)
    // 缺陷(缺墨/飞墨/拖尾)，默认 0 关闭
    double missRate = 0.0, splashRate = 0.0, smearRate = 0.0;

    // —— 融合墨色 ——
    cv::Scalar inkColor = cv::Scalar(0, 0, 0);

    // —— 标注 & 输出 ——
    bool   genYolo   = true;          // 是否同时生成 yolo 标签(功能2)
    int    boxPad    = 2;             // 数字最小外接矩形向外扩充像素
    int    nameWidth = 6;             // 文件名编号宽度(6 -> 000000)
    std::string imgExt = "png";       // 图片扩展名(不带点)，建议 png 保真
};

// 读取 txt：每行 -> 一个样本；样本内容 = 行内空格分隔的数字串
// 失败(文件打不开/没有有效行)返回 false
bool read_txt_lines(const std::string& txtPath,
                    std::vector<std::vector<std::string>>& samples);

// 收集背景文件夹内的图片(png/jpg/jpeg/bmp，按文件名排序)
bool collect_backgrounds(const std::string& bgDir,
                         std::vector<std::string>& bgFiles);

// 核心接口：功能1 + 功能2。
//   txtPath : 喷码文本文件
//   bgDir   : 背景图片文件夹(循环使用)
//   outRoot : 输出根目录，自动创建 images/ 与 labels/
// 返回：>=0 成功保存的图片数；-1 文本读取失败；-2 背景目录无可用图片。
int batch_generate(const std::string& txtPath,
                   const std::string& bgDir,
                   const std::string& outRoot,
                   const BatchConfig& cfg = BatchConfig());

} // namespace penma_batch

#endif // PENMA_BATCH_H
