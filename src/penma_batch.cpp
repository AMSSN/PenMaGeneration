#include "penma_batch.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace penma_batch
{

//------------------------------------------------------------------------------
// 文本读取：每行一个样本，行内数字串用空格分隔
//------------------------------------------------------------------------------
bool read_txt_lines(const std::string& txtPath,
                    std::vector<std::vector<std::string>>& samples)
{
    std::ifstream in(txtPath, std::ios::in);
    if (!in.is_open())
        return false;

    samples.clear();
    std::string line;
    while (std::getline(in, line))
    {
        // 去掉首尾空白(含 \r)
        const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
        auto it = std::find_if(line.begin(), line.end(),
            [&](char c) { return notSpace(static_cast<unsigned char>(c)); });
        line.erase(line.begin(), it);
        it = std::find_if(line.rbegin(), line.rend(),
            [&](char c) { return notSpace(static_cast<unsigned char>(c)); }).base();
        line.erase(it, line.end());
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::vector<std::string> toks;
        std::string t;
        while (iss >> t)
            toks.push_back(t);
        if (!toks.empty())
            samples.push_back(std::move(toks));
    }
    return !samples.empty();
}

//------------------------------------------------------------------------------
// 背景收集(png/jpg/jpeg/bmp)
//------------------------------------------------------------------------------
bool collect_backgrounds(const std::string& bgDir,
                         std::vector<std::string>& bgFiles)
{
    std::error_code ec;
    if (!fs::exists(bgDir, ec) || !fs::is_directory(bgDir, ec))
        return false;

    std::vector<std::string> files;
    for (const fs::directory_entry& de : fs::directory_iterator(bgDir, ec))
    {
        if (!de.is_regular_file(ec))
            continue;
        std::string ext = de.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            files.push_back(de.path().string());
    }
    std::sort(files.begin(), files.end());
    bgFiles.swap(files);
    return !bgFiles.empty();
}

namespace
{

std::string idx_to_name(size_t i, int width)
{
    std::ostringstream os;
    os << std::setw(max(1, width)) << std::setfill('0') << i;
    return os.str();
}

// 只保留数字字符(与 DotNumber::set_numbers 内部清洗规则一致)
std::string digits_only(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c >= '0' && c <= '9')
            out.push_back(c);
    return out;
}

// 用仿射 map 把 mask 坐标矩形变换到旋转画布 -> 平移到 loc -> 外扩 pad -> 裁进图内
bool to_pixel_box(const cv::Rect& r, const cv::Mat& map, const cv::Point& loc,
                  int pad, const cv::Size& imSz, cv::Rect& outBox)
{
    const double* m0 = map.ptr<double>(0);
    const double* m1 = map.ptr<double>(1);
    const double X[4] = { (double)r.x, (double)(r.x + r.width),
                          (double)r.x, (double)(r.x + r.width) };
    const double Y[4] = { (double)r.y, (double)r.y,
                          (double)(r.y + r.height), (double)(r.y + r.height) };

    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (int i = 0; i < 4; ++i)
    {
        const double fx = m0[0] * X[i] + m0[1] * Y[i] + m0[2] + loc.x;
        const double fy = m1[0] * X[i] + m1[1] * Y[i] + m1[2] + loc.y;
        x0 = min(x0, fx);  x1 = max(x1, fx);
        y0 = min(y0, fy);  y1 = max(y1, fy);
    }

    const int ix0 = (int)std::floor(x0) - pad;
    const int iy0 = (int)std::floor(y0) - pad;
    const int ix1 = (int)std::ceil(x1) + pad;
    const int iy1 = (int)std::ceil(y1) + pad;

    outBox = cv::Rect(ix0, iy0, ix1 - ix0, iy1 - iy0) &
             cv::Rect(0, 0, imSz.width, imSz.height);
    return outBox.width > 0 && outBox.height > 0;
}

// 预计算旋转后画布尺寸(与 DotNumber 内部 do_paste 的公式保持一致)
cv::Size rotated_size(int W, int H, double angleDeg)
{
    if (std::abs(angleDeg) <= 1e-3)
        return cv::Size(W, H);
    const double rad = angleDeg * CV_PI / 180.0;
    const double ca = std::abs(std::cos(rad));
    const double sa = std::abs(std::sin(rad));
    return cv::Size(static_cast<int>(std::ceil(H * sa + W * ca)),
                    static_cast<int>(std::ceil(H * ca + W * sa)));
}

} // namespace

//------------------------------------------------------------------------------
// 批量生成主入口
//------------------------------------------------------------------------------
int batch_generate(const std::string& txtPath,
                   const std::string& bgDir,
                   const std::string& outRoot,
                   const BatchConfig& cfg)
{
    std::vector<std::vector<std::string>> samples;
    if (!read_txt_lines(txtPath, samples))
        return -1;

    std::vector<std::string> bgFiles;
    if (!collect_backgrounds(bgDir, bgFiles))
        return -2;

    // 读入全部背景(循环使用)，读取失败自动跳过
    std::vector<cv::Mat> bgs;
    bgs.reserve(bgFiles.size());
    for (const std::string& f : bgFiles)
    {
        cv::Mat m = cv::imread(f, cv::IMREAD_COLOR);
        if (!m.empty())
            bgs.push_back(m);
    }
    if (bgs.empty())
        return -2;

    // 输出目录
    const std::string imgDir = (fs::path(outRoot) / "images").string();
    const std::string lblDir = (fs::path(outRoot) / "labels").string();
    {
        std::error_code ec;
        fs::create_directories(imgDir, ec);
        if (cfg.genYolo)
            fs::create_directories(lblDir, ec);
    }

    cv::RNG rng(cv::getTickCount());
    DotNumber dn;
    int saved = 0;

    for (size_t si = 0; si < samples.size(); ++si)
    {
        // ---- 该样本内容：多个数字串纵向堆叠成多行喷码 ----
        std::vector<std::string> rows;
        rows.reserve(samples[si].size());
        for (const std::string& tok : samples[si])
        {
            const std::string c = digits_only(tok);
            if (!c.empty())
                rows.push_back(c);
        }
        if (rows.empty())
            continue;

        // ---- 背景：每行文本对应一张背景图，图片循环使用 ----
        const cv::Mat& bg = bgs[si % bgs.size()];

        // ---- 生成点阵喷码掩码与风格后处理 ----
        dn.set_numbers(rows);
        dn.dotSize    = cfg.dotSize;
        dn.dotSpac    = cfg.dotSpac;
        dn.numSpac    = cfg.numSpac;
        dn.lineSpac   = cfg.lineSpac;
        dn.sideSpac   = cfg.sideSpac;
        dn.sideSpacX  = cfg.sideSpacX;
        dn.maskRotate = cfg.maskRotate;

        dn.gen_mask_mat();
        if (cfg.driftPix > 0)
            dn.add_style_drift(cfg.driftPix, cfg.driftConf);
        if (cfg.filterK > 0)
            dn.add_style_filter(cfg.filterK);
        if (cfg.missRate > 0.0 || cfg.splashRate > 0.0 || cfg.smearRate > 0.0)
            dn.add_style_defect(cfg.missRate, cfg.splashRate, cfg.smearRate);

        const cv::Mat alpha = dn.get_mask_result(true);   // 墨点 alpha
        if (alpha.empty())
            continue;
        const int W = alpha.cols, H = alpha.rows;

        // ---- 随机旋转角度(与内部贴图一致，避免画布尺寸估算偏差) ----
        double ang = 0.0;
        if (cfg.maskRotate > 0.0)
            ang = rng.uniform(-cfg.maskRotate, cfg.maskRotate);
        if (std::abs(ang) <= 1e-3)
            ang = 0.0;
        const cv::Size canvas = rotated_size(W, H, ang);

        // ---- 左上 1/4 区域内随机定位(整块喷码画布保持可见) ----
        const int qW = max(1, bg.cols / 2);
        const int qH = max(1, bg.rows / 2);
        const int maxX = max(0, qW - canvas.width);
        const int maxY = max(0, qH - canvas.height);
        const cv::Point loc(rng.uniform(0, maxX + 1),
                            rng.uniform(0, maxY + 1));

        // ---- 融合 ----
        DotNumber::PasteInfo info;
        const cv::Mat fused = dn.paste_mask(bg, loc, cfg.inkColor, ang, &info);
        if (fused.empty())
            continue;

        const std::string stem = idx_to_name(saved, cfg.nameWidth);
        const std::string imgPath =
            (fs::path(imgDir) / (stem + "." + cfg.imgExt)).string();
        if (!cv::imwrite(imgPath, fused))
        {
            std::cerr << "[penma_batch] 保存图片失败: " << imgPath << "\n";
            continue;
        }
        ++saved;

        if (!cfg.genYolo)
            continue;

        // ---- YOLO 标签：每个数字一个框，框外扩 boxPad 像素 ----
        const std::vector<std::vector<cv::Rect>> rects = dn.get_digit_rects();
        std::ostringstream lbl;
        const int pad = max(0, cfg.boxPad);
        for (size_t li = 0; li < rects.size() && li < rows.size(); ++li)
        {
            const std::string& text = rows[li];
            const std::vector<cv::Rect>& lineRects = rects[li];
            for (size_t k = 0; k < lineRects.size() && k < text.size(); ++k)
            {
                if (lineRects[k].width <= 0 || lineRects[k].height <= 0)
                    continue;
                cv::Rect box;
                if (!to_pixel_box(lineRects[k], info.mapToRot, loc, pad,
                                  fused.size(), box))
                    continue;
                const float cx = static_cast<float>((box.x + box.width / 2.0) / fused.cols);
                const float cy = static_cast<float>((box.y + box.height / 2.0) / fused.rows);
                const float bw = static_cast<float>(box.width) / fused.cols;
                const float bh = static_cast<float>(box.height) / fused.rows;
                lbl << static_cast<int>(text[k] - '0') << ' '
                    << std::fixed << std::setprecision(6)
                    << cx << ' ' << cy << ' ' << bw << ' ' << bh << '\n';
            }
        }

        const std::string lblPath = (fs::path(lblDir) / (stem + ".txt")).string();
        std::ofstream of(lblPath, std::ios::out | std::ios::trunc);
        if (of.is_open())
            of << lbl.str();
    }

    return saved;
}

} // namespace penma_batch
