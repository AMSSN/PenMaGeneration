#ifndef PENMA_H
#define PENMA_H
#include <vector>
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <filesystem>
#include <algorithm>
// 3rdpartys
#include "opencv.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include "halconcpp/HalconCpp.h"
using namespace HalconCpp;
using namespace cv;


// 10 个数字，每个 7 行 × 5 列
using Digit = std::array<std::array<float, 5>, 7>;


class DotNumber {
public:
    int dotSize = 5;       //喷码大小（单个墨点的直径，建议取奇数）
    int dotSpac = 1;       //墨点间距（点阵内相邻墨点之间的间隙）
    int numSpac = 3;       //字符间距
    int sideSpac = 2;      //上下边距
    int sideSpacX = 0;     //左右边距（默认0，需要留白时再设）
    bool antiAlias = false;//墨点是否抗锯齿（true 时输出不再是严格二值）
    int lineSpac = 2;      //行距（多行喷码时，相邻两行之间的间隔）
    float maskRotate = 1.2;// 整个mask在贴图时的旋转角度范围(随机旋转正负1.2°)
    cv::Scalar color = cv::Scalar(0, 0, 0);

    // Get/Set方法
    std::string get_numbers();                          // 多行时以 '\n' 分隔
    void set_numbers(std::string nums);                 // 单行
    void set_numbers(std::vector<int> nums);            // 单行（数字序列）
    void set_numbers(std::vector<std::string> nums);    // 多行，每个元素一行
    cv::Mat get_mask_result(bool alpha=true); //获取内部变量m_mask或者是m_alpha_mask
    // 渲染成cv::Mat图片
    cv::Mat gen_mask_mat();   //掩码图: 背景=m_background(白255), 墨点=m_foreground(黑0)
    cv::Mat gen_alpha_mat();  //alpha图: 墨点=255, 其余=0, 可直接用于 copyTo 贴图
    // 图片处理，为了模仿喷码打印出来后再拍摄的效果。
    // 注意: conf 原声明为 int，默认值 0.95 会被截断成 0（等于永远不漂移），已改成 double
    void add_style_drift(int pix=2, double conf=0.95); // 95%概率，喷码中心点会随机偏移2个像素。
    void add_style_filter(int core_size); //滤波以模糊边界
    // 喷码缺陷: 缺墨(局部墨量缺失/变淡)、飞墨(墨点周围飞溅小点)、拖尾(沿打印方向拉出淡尾)
    void add_style_defect(double missRate=0.05, double splashRate=0.03, double smearRate=0.10);
    // 贴图和贴图后处理
    // 背景支持 CV_8UC1 / CV_8UC3 / CV_8UC4；墨色默认取成员变量 color(BGR)
    cv::Mat paste_mask(cv::Mat bg, cv::Point loc); //检查位置，然后将喷码图片覆盖在背景上，并做一定的后处理让融合效果不那么生硬。
    cv::Mat paste_mask(cv::Mat bg, cv::Point loc, const cv::Scalar& inkColor); // 指定墨色（覆盖成员变量 color）

    // —— 批量合成 / 数据集标注支持 ——
    // 旋转/贴图信息：供外部把 mask 内的标注框精确变换到最终融合图上
    struct PasteInfo {
        double   angleDeg = 0.0;   // 本次实际旋转角度(度)
        cv::Size rotCanvas;        // 旋转后的透明画布尺寸
        cv::Mat  mapToRot;         // 2x3 仿射矩阵：mask像素坐标 -> 旋转画布像素坐标
    };
    // 以指定角度(度)旋转后贴图(不再用 maskRotate 随机)；info 可选返回旋转信息用于生成标注
    cv::Mat paste_mask(cv::Mat bg, cv::Point loc, const cv::Scalar& inkColor,
                       double fixedAngleDeg, PasteInfo* info = nullptr);

    // 当前掩码中每个数字的最小外接矩形(mask 坐标系，逐行逐字符)。
    // 顺序与 set_numbers 的“行-字符”一致；该字符无墨迹时返回空矩形(width<=0)。
    std::vector<std::vector<cv::Rect>> get_digit_rects() const;

private:
    std::string m_NumberStr;
    // 不管是 string / vector<int> / vector<string>，统一转换成内部定义的结构 m_DigitLines
    // 每个元素是一行，行内每个元素是一个字符的点阵
    std::vector<std::vector<Digit>> m_DigitLines;
    cv::Mat m_mask;
    cv::Mat m_alpha_mask;


    uchar m_background = 255; //
    uchar m_foreground = 0;   //

    const std::array<Digit, 10> NumFont = { {
            // 0
            {{{0,1,1,1,0},{1,0,0,0,1},{1,0,0,0,1},{1,0,0,0,1},{1,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0}}},
            // 1
            {{{0,0,1,0,0},{0,1,1,0,0},{1,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1}}},
            // 2
            {{{0,1,1,1,0},{1,0,0,0,1},{0,0,0,0,1},{0,0,0,1,0},{0,0,1,0,0},{0,1,0,0,0},{1,1,1,1,1}}},
            // 3
            {{{0,1,1,1,0},{1,0,0,0,1},{0,0,0,0,1},{0,0,1,1,0},{0,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0}}},
            // 4
            {{{0,0,0,1,0},{0,0,1,1,0},{0,1,0,1,0},{1,0,0,1,0},{1,1,1,1,1},{0,0,0,1,0},{0,0,0,1,0}}},
            // 5
            {{{1,1,1,1,1},{1,0,0,0,0},{1,1,1,1,0},{0,0,0,0,1},{0,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0}}},
            // 6
            {{{0,0,1,1,0},{0,1,0,0,0},{1,0,0,0,0},{1,1,1,1,0},{1,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0}}},
            // 7
            {{{1,1,1,1,1},{0,0,0,0,1},{0,0,0,1,0},{0,0,1,0,0},{0,1,0,0,0},{0,1,0,0,0},{0,1,0,0,0}}},
            // 8
            {{{0,1,1,1,0},{1,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0},{1,0,0,0,1},{1,0,0,0,1},{0,1,1,1,0}}},
            // 9
            {{{0,1,1,1,0},{1,0,0,0,1},{1,0,0,0,1},{0,1,1,1,1},{0,0,0,0,1},{0,0,0,1,0},{0,1,1,0,0}}}
        } };
    cv::Point m_fun_drift(cv::Point ori_loc, int pix); // 随机偏移的策略，上下左右、斜向上下左右共8个方向。

    // 最近一次重绘(gen_mask_mat / add_style_drift)记录的内容布局, 供 get_digit_rects 定位字符格子
    int  m_layoutOffX = 0;
    int  m_layoutOffY = 0;
    int  m_layoutContW = 0;
    int  m_layoutContH = 0;
    int  m_layoutNumSpac = 0;
    int  m_layoutLineSpac = 0;
    bool m_layoutValid = false;
    void record_layout(int offX, int offY, int contW, int contH,
                       int numSpac, int lineSpac);
};




#endif // DOTNUMBER_H
