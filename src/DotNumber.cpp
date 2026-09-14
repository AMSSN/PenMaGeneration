#include "DotNumber.h"



namespace {

// 点阵尺寸常量
constexpr int FH = 7;   // 点阵行数
constexpr int FW = 5;   // 点阵列数

// 掩码的几何参数（所有参数在此统一做下界保护，避免多处口径不一致）
struct MaskGeom
{
    int dotSize = 1;
    int dotSpac = 0;
    int numSpac = 0;
    int lineSpac = 0;
    int sideSpac = 0;
    int sideSpacX = 0;

    int dotStep = 1;    // 点阵步进（相邻墨点圆心距）
    int contentW = 0;   // 单个字符的内容宽
    int contentH = 0;   // 单个字符的内容高

    int nLines = 0;     // 行数
    int maxLineW = 0;   // 最宽那一行的宽（各行左对齐）
    int imgW = 0;       // 画布宽
    int imgH = 0;       // 画布高
};

MaskGeom calc_geom(const std::vector<std::vector<Digit>>& lines,
    int dotSize, int dotSpac, int numSpac, int lineSpac, int sideSpac, int sideSpacX)
{
    MaskGeom g;
    g.dotSize = max(1, dotSize);
    g.dotSpac = max(0, dotSpac);
    g.numSpac = max(0, numSpac);
    g.lineSpac = max(0, lineSpac);
    g.sideSpac = max(0, sideSpac);
    g.sideSpacX = max(0, sideSpacX);

    g.dotStep = g.dotSize + g.dotSpac;
    g.contentW = FW * g.dotStep - g.dotSpac;
    g.contentH = FH * g.dotStep - g.dotSpac;

    // 逐行计算宽度，画布宽取最宽的一行
    g.nLines = static_cast<int>(lines.size());
    for (const std::vector<Digit>& line : lines)
    {
        int w = 0;
        if (!line.empty())
            w = static_cast<int>(line.size()) * g.contentW
              + (static_cast<int>(line.size()) - 1) * g.numSpac;
        g.maxLineW = max(g.maxLineW, w);
    }

    g.imgW = g.maxLineW + 2 * g.sideSpacX;
    g.imgH = g.nLines * g.contentH + (g.nLines - 1) * g.lineSpac + 2 * g.sideSpac;
    return g;
}

/**
 * 计算所有墨点的圆心坐标（支持多行，行内左对齐）
 * @param offX/offY  整体原点偏移（gen_mask_mat 用边距；add_style_drift 用边距+漂移余量）
 */
void collect_dot_centers(const std::vector<std::vector<Digit>>& lines, const MaskGeom& g,
    int offX, int offY, std::vector<cv::Point>& centers)
{
    const int r = g.dotSize / 2;
    centers.clear();
    centers.reserve(lines.size() * 16 * FW * FH);

    for (size_t li = 0; li < lines.size(); ++li)
    {
        const int y0 = offY + static_cast<int>(li) * (g.contentH + g.lineSpac);
        const std::vector<Digit>& line = lines[li];

        for (size_t i = 0; i < line.size(); ++i)
        {
            const int x0 = offX + static_cast<int>(i) * (g.contentW + g.numSpac);
            const Digit& p = line[i];
            for (int y = 0; y < FH; ++y)
            {
                for (int x = 0; x < FW; ++x)
                {
                    if (p[y][x] > 0.5f)
                        centers.emplace_back(x0 + x * g.dotStep + r,
                                             y0 + y * g.dotStep + r);
                }
            }
        }
    }
}

// 按圆心列表绘制墨点
void draw_dots(cv::Mat& img, const std::vector<cv::Point>& centers,
    int radius, const cv::Scalar& value, int lineType)
{
    for (const cv::Point& c : centers)
        cv::circle(img, c, radius, value, cv::FILLED, lineType);
}

// 由 m_mask 同步生成 m_alpha_mask
// alpha = 255 * (mask - background) / (foreground - background)
void update_alpha(const cv::Mat& mask, cv::Mat& alpha, uchar fgValue, uchar bgValue)
{
    if (mask.empty())
    {
        alpha = cv::Mat();
        return;
    }

    const double denom = static_cast<double>(static_cast<int>(fgValue) -
                                             static_cast<int>(bgValue));
    if (std::abs(denom) < 1e-6)   // 前景与背景同值，掩码无意义
    {
        alpha = cv::Mat::zeros(mask.size(), CV_8UC1);
        return;
    }

    cv::Mat f;
    mask.convertTo(f, CV_32F, 1.0, -static_cast<double>(bgValue));
    f.convertTo(alpha, CV_8U, 255.0 / denom);
}

// 由 alpha 反推 mask：mask = background + a * (foreground - background)
void update_mask_from_alpha(const cv::Mat& a, cv::Mat& mask, uchar fgValue, uchar bgValue)
{
    if (a.empty())
    {
        mask = cv::Mat();
        return;
    }
    const double denom = static_cast<double>(static_cast<int>(fgValue) -
                                             static_cast<int>(bgValue));
    cv::Mat maskF;
    a.convertTo(maskF, CV_32F, denom, static_cast<double>(bgValue));
    maskF.convertTo(mask, CV_8U);
}

/**
 * 贴图核心：正片叠底混合
 * result = bg * (1 - a) + (bg ⊙ ink / 255) * a
 * 支持 CV_8UC1 / CV_8UC3 / CV_8UC4 背景
 */
cv::Mat do_paste(const cv::Mat& mask, const cv::Mat& alphaMask, const cv::Mat& bg,
                 cv::Point loc, const cv::Scalar& inkColor, float rotRange,
                 DotNumber::PasteInfo* pasteInfo = nullptr,   // 可选，返回旋转信息
                 bool useFixedAngle = false, double fixedAngleDeg = 0.0)
{
    if (bg.empty())
        return bg.clone();
    if (bg.depth() != CV_8U)          // 目前只支持 8bit 背景
        return bg.clone();
    if (mask.empty() || alphaMask.empty())
        return bg.clone();
    static cv::RNG rng(cv::getTickCount());

    // ---------- 0. 整体旋转（默认 ±rotRange 随机；也支持外部固定角度） ----------
    cv::Mat alphaRot = alphaMask;
    double angleDeg = 0.0;
    cv::Size rotCanvas = alphaMask.size();
    cv::Mat mapToRot = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0,
                                                       0.0, 1.0, 0.0);
    if (std::abs(rotRange) > 1e-3f || useFixedAngle)
    {
        const double angle = useFixedAngle
            ? fixedAngleDeg
            : rng.uniform(-(double)rotRange, (double)rotRange);
        if (std::abs(angle) > 1e-3)
        {
            angleDeg = angle;
            const cv::Point2f c(alphaMask.cols / 2.0, alphaMask.rows / 2.0);
            cv::Mat M = cv::getRotationMatrix2D(c, angle, 1.0);

            // 旋转后的外接矩形尺寸，保证内容不被裁掉
            const double rad = angle * CV_PI / 180.0;
            const double ca = std::abs(std::cos(rad));
            const double sa = std::abs(std::sin(rad));
            const int newW = cvCeil(alphaMask.rows * sa + alphaMask.cols * ca);
            const int newH = cvCeil(alphaMask.rows * ca + alphaMask.cols * sa);

            // 补偿平移，让旋转后的内容居中落在新的外接矩形里
            M.at<double>(0, 2) += newW / 2.0 - c.x;
            M.at<double>(1, 2) += newH / 2.0 - c.y;

            // 边界填 0（透明），线性插值顺带得到抗锯齿的软边
            cv::warpAffine(alphaMask, alphaRot, M, cv::Size(newW, newH),
                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

            rotCanvas = cv::Size(newW, newH);
            mapToRot = M.clone();
        }
    }
    if (pasteInfo)
    {
        pasteInfo->angleDeg = angleDeg;
        pasteInfo->rotCanvas = rotCanvas;
        pasteInfo->mapToRot = mapToRot.clone();
    }

    // ---------- 1. 位置检查：取掩码与背景的交集，允许部分出界 ----------
    const cv::Rect bgRect(0, 0, bg.cols, bg.rows);
    const cv::Rect maskRect(loc.x, loc.y, alphaRot.cols, alphaRot.rows);
    const cv::Rect inter = maskRect & bgRect;
    if (inter.width <= 0 || inter.height <= 0)   // 完全在画面外
        return bg.clone();

    // 掩码上对应的子区域
    const cv::Rect maskSub(inter.x - maskRect.x, inter.y - maskRect.y,
                           inter.width, inter.height);

    const int cn = bg.channels();
    cv::Mat dst = bg.clone();
    cv::Mat roi = dst(inter);

    // ---------- 2. alpha：墨点=1，空白=0，边缘为过渡值 ----------
    cv::Mat a1;
    alphaRot(maskSub).convertTo(a1, CV_32F, 1.0 / 255.0);


    // 墨量浓度：整串喷码随机偏淡，模拟喷墨量不足
    const double density = rng.uniform(0.85, 1.0);
    // 墨色浓淡不均：只在有墨处做【乘性】扰动。
    // 注意:不能写成 a1 = a1 + noise —— 加性噪声会让掩码的空白区也带上 alpha,
    // 导致整块喷码区域被墨色染成一片（表现为发暗 / 偏色）
    cv::Mat noise(a1.size(), CV_32F);
    rng.fill(noise, cv::RNG::NORMAL, 1.0, 0.06);
    a1 = a1.mul(noise) * density;
    a1.setTo(0.0f, a1 < 0.0f);
    a1.setTo(1.0f, a1 > 1.0f);

    cv::Mat inkMask = (a1 > 0.0f); // 硬掩膜
    // 扩展到与背景相同的通道数
    cv::Mat aN;
    cv::merge(std::vector<cv::Mat>(cn, a1), aN);

    // ---------- 3. 墨色（BGR），按背景通道数适配 ----------
    cv::Scalar inkScalar = inkColor;
    if (cn == 1)   // 灰度背景：取墨色的亮度
        inkScalar = cv::Scalar((inkColor[0] + inkColor[1] + inkColor[2]) / 3.0);
    else if (cn == 4)
        inkScalar[3] = 255.0;   // 保留背景原有 alpha

    // ---------- 4. 正片叠底混合 ----------
    cv::Mat bgf;
    roi.convertTo(bgf, CV_32FC(cn));

    cv::Mat ink(inter.size(), CV_32FC(cn), inkScalar);
    cv::Mat multiplied = bgf.mul(ink) / 255.0;

    cv::Mat diff = multiplied - bgf;
    cv::Mat resf = bgf + diff.mul(aN);

    cv::Mat patch;
    resf.convertTo(patch, roi.type());
    patch.copyTo(roi, inkMask);             


    return dst;
}

} // namespace


void DotNumber::record_layout(int offX, int offY, int contW, int contH,
                              int numSpac, int lineSpac)
{
    m_layoutOffX = offX;
    m_layoutOffY = offY;
    m_layoutContW = contW;
    m_layoutContH = contH;
    m_layoutNumSpac = numSpac;
    m_layoutLineSpac = lineSpac;
    m_layoutValid = true;
}


//------------------------------------------------------------------------------
// Get / Set
//------------------------------------------------------------------------------

std::string DotNumber::get_numbers()
{
    return m_NumberStr;
}


void DotNumber::set_numbers(std::string nums)
{
    m_NumberStr.clear();
    m_DigitLines.clear();

    std::vector<Digit> line;
    line.reserve(nums.size());

    for (char c : nums)
    {
        if (c >= '0' && c <= '9')
        {
            line.push_back(NumFont[c - '0']);
            m_NumberStr.push_back(c);
        }
    }

    if (!line.empty())
        m_DigitLines.push_back(line);
    else
        m_NumberStr.clear();
}


void DotNumber::set_numbers(std::vector<int> nums)
{
    m_NumberStr.clear();
    m_DigitLines.clear();

    std::vector<Digit> line;
    line.reserve(nums.size());

    for (int d : nums)
    {
        if (d >= 0 && d <= 9)
        {
            line.push_back(NumFont[d]);
            m_NumberStr.push_back(static_cast<char>('0' + d));
        }
    }

    if (!line.empty())
        m_DigitLines.push_back(line);
    else
        m_NumberStr.clear();
}


// 多行输入：每个元素一行，空行自动跳过
void DotNumber::set_numbers(std::vector<std::string> nums)
{
    m_NumberStr.clear();
    m_DigitLines.clear();
    m_DigitLines.reserve(nums.size());

    std::vector<std::string> cleaned;   // 只保留数字字符后的行文本

    for (const std::string& s : nums)
    {
        std::vector<Digit> line;
        std::string cs;
        line.reserve(s.size());
        cs.reserve(s.size());

        for (char c : s)
        {
            if (c >= '0' && c <= '9')
            {
                line.push_back(NumFont[c - '0']);
                cs.push_back(c);
            }
        }

        if (line.empty())   // 跳过空行（不含任何有效数字）
            continue;

        m_DigitLines.push_back(line);
        cleaned.push_back(cs);
    }

    for (size_t i = 0; i < cleaned.size(); ++i)
    {
        if (i > 0)
            m_NumberStr.push_back('\n');
        m_NumberStr += cleaned[i];
    }
}


cv::Mat DotNumber::get_mask_result(bool alpha)
{
    // 尚未生成时先生成一次，避免返回空图
    if (m_mask.empty())
        gen_mask_mat();

    if (alpha)
    {
        if (m_alpha_mask.empty())
            update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);
        return m_alpha_mask;
    }
    return m_mask;
}


//------------------------------------------------------------------------------
// 生成掩码图（单行 / 多行）
//------------------------------------------------------------------------------

cv::Mat DotNumber::gen_mask_mat()
{
    m_mask = cv::Mat();
    m_alpha_mask = cv::Mat();

    if (m_DigitLines.empty())
        return m_mask;

    const MaskGeom g = calc_geom(m_DigitLines, dotSize, dotSpac, numSpac,
                                 lineSpac, sideSpac, sideSpacX);

    // 记录当前布局，供 get_digit_rects 逐字符定位
    record_layout(g.sideSpacX, g.sideSpac, g.contentW, g.contentH,
                  g.numSpac, g.lineSpac);

    // 背景填充 m_background（默认 255 白）
    m_mask = cv::Mat(g.imgH, g.imgW, CV_8UC1, cv::Scalar(m_background));

    // 计算墨点圆心并绘制
    std::vector<cv::Point> centers;
    collect_dot_centers(m_DigitLines, g, g.sideSpacX, g.sideSpac, centers);

    draw_dots(m_mask, centers, g.dotSize / 2, cv::Scalar(m_foreground),
              antiAlias ? cv::LINE_AA : cv::LINE_8);

    update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);
    return m_mask;
}


cv::Mat DotNumber::gen_alpha_mat()
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_alpha_mask.empty())
        update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);

    return m_alpha_mask;
}


//------------------------------------------------------------------------------
// 随机漂移策略：上下左右 + 四个斜向，共 8 个方向
//------------------------------------------------------------------------------

cv::Point DotNumber::m_fun_drift(cv::Point ori_loc, int pix)
{
    if (pix <= 0)
        return ori_loc;

    // 8 个方向对应的 (dx, dy)
    static const int DX[8] = {  0,  0, -1,  1, -1,  1, -1,  1 };
    static const int DY[8] = { -1,  1,  0,  0, -1, -1,  1,  1 };

    // 函数内静态 RNG：只播种一次，避免高频调用时重复播种得到相同序列
    static cv::RNG rng(cv::getTickCount());

    const int dir = rng.uniform(0, 8);        // [0, 8)
    const int mag = rng.uniform(1, pix + 1);  // [1, pix]

    return cv::Point(ori_loc.x + DX[dir] * mag,
                     ori_loc.y + DY[dir] * mag);
}


//------------------------------------------------------------------------------
// 后处理 1：墨点位置漂移
//------------------------------------------------------------------------------

void DotNumber::add_style_drift(int pix, double conf)
{
    if (m_DigitLines.empty())
        return;

    pix = std::abs(pix);
    if (pix <= 0)
        return;

    conf = min(1.0, max(0.0, conf));

    const MaskGeom g = calc_geom(m_DigitLines, dotSize, dotSpac, numSpac,
                                 lineSpac, sideSpac, sideSpacX);

    // 四周预留 pix 像素，保证漂移后的墨点不会被画布裁掉（重复调用结果一致）
    const int offX = g.sideSpacX + pix;
    const int offY = g.sideSpac + pix;

    std::vector<cv::Point> centers;
    collect_dot_centers(m_DigitLines, g, offX, offY, centers);

    // 每个墨点以 conf 的概率发生随机漂移
    static cv::RNG rng(cv::getTickCount());
    for (cv::Point& c : centers)
    {
        if (rng.uniform(0.0, 1.0) < conf)
            c = m_fun_drift(c, pix);
    }

    // 重绘（漂移是几何层面的，只能重新生成，会覆盖此前的滤波效果，故建议先 drift 再 filter）
    m_mask = cv::Mat(g.imgH + 2 * pix, g.imgW + 2 * pix, CV_8UC1, cv::Scalar(m_background));
    draw_dots(m_mask, centers, g.dotSize / 2, cv::Scalar(m_foreground),
              antiAlias ? cv::LINE_AA : cv::LINE_8);

    // 漂移会整体扩展画布并改变内容原点，重新记录布局
    record_layout(offX, offY, g.contentW, g.contentH, g.numSpac, g.lineSpac);

    update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);
}


//------------------------------------------------------------------------------
// 后处理 2：滤波模糊边界
//------------------------------------------------------------------------------

void DotNumber::add_style_filter(int core_size)
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_mask.empty())
        return;

    // 高斯核必须是正奇数，sigma=0 时由核尺寸自动推导
    int k = max(3, std::abs(core_size));
    if (k % 2 == 0)
        k += 1;

    cv::Mat blurred;
    cv::GaussianBlur(m_mask, blurred, cv::Size(k, k), 0, 0);
    m_mask = blurred;

    update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);
}


//------------------------------------------------------------------------------
// 后处理 3：喷码缺陷
//------------------------------------------------------------------------------

void DotNumber::add_style_defect(double missRate, double splashRate, double smearRate)
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_mask.empty() || m_alpha_mask.empty())
        return;

    static cv::RNG rng(cv::getTickCount());

    const int W = m_alpha_mask.cols;
    const int H = m_alpha_mask.rows;
    const int baseDot = max(1, dotSize);
    const int dotStepV = baseDot + max(0, dotSpac);

    // 统一在 alpha 域 [0,1] 上操作（墨点=1，空白=0）
    cv::Mat a;
    m_alpha_mask.convertTo(a, CV_32F, 1.0 / 255.0);

    // ---------------- 1. 缺墨：随机斑块把局部墨量"喷空/喷淡" ----------------
    missRate = min(1.0, max(0.0, missRate));
    if (missRate > 0.0)
    {
        // 缺陷斑块数量与墨点总数挂钩
        const int dotCount = max(1, (W * H) / (dotStepV * dotStepV));
        const int patchCount = max(1, cvRound(missRate * dotCount));

        cv::Mat reduce = cv::Mat::zeros(a.size(), CV_32F);
        for (int i = 0; i < patchCount; ++i)
        {
            cv::Point c(rng.uniform(0, W), rng.uniform(0, H));
            int r = rng.uniform(1, max(2, baseDot / 2 + 1));
            double keep = rng.uniform(0.0, 0.35);   // 保留的墨量，越小缺得越狠
            cv::circle(reduce, c, r, cv::Scalar(1.0 - keep), cv::FILLED);
        }

        // 乘法衰减，保证空白处不会被"凭空造墨"
        cv::Mat keepMat = cv::Mat::ones(a.size(), CV_32F) - reduce;
        a = a.mul(keepMat);
    }

    // ---------------- 2. 飞墨：墨点周围随机飞溅小点 ----------------
    splashRate = min(1.0, max(0.0, splashRate));
    if (splashRate > 0.0)
    {
        cv::Mat inkBin;
        cv::threshold(m_alpha_mask, inkBin, 128, 255, cv::THRESH_BINARY);

        std::vector<cv::Point> inkPts;
        cv::findNonZero(inkBin, inkPts);

        if (!inkPts.empty())
        {
            const int splashCount = max(1,
                cvRound(splashRate * static_cast<int>(inkPts.size()) / 20.0));

            for (int i = 0; i < splashCount; ++i)
            {
                const cv::Point& src = inkPts[rng.uniform(0, static_cast<int>(inkPts.size()))];

                const double ang = rng.uniform(0.0, CV_2PI);
                const double dist = rng.uniform(baseDot * 0.6, baseDot * 2.0);
                cv::Point p(cvRound(src.x + std::cos(ang) * dist),
                            cvRound(src.y + std::sin(ang) * dist));

                const int r = rng.uniform(1, max(2, baseDot / 3));
                const double v = rng.uniform(0.25, 0.9);   // 飞溅点的浓度
                cv::circle(a, p, r, cv::Scalar(v), cv::FILLED);
            }
        }
    }

    // ---------------- 3. 拖尾：某一条水平带沿打印方向被拉花 ----------------
    smearRate = min(1.0, max(0.0, smearRate));
    if (smearRate > 0.0)
    {
        const int bandH = max(1, cvRound(H * rng.uniform(0.15, 0.5)));
        const int bandY = rng.uniform(0, max(1, H - bandH));
        cv::Mat band = a(cv::Rect(0, bandY, W, bandH));

        const int tailLen = max(1, cvRound(smearRate * W));
        const bool toRight = (rng.uniform(0, 2) == 0);

        // 单边指数衰减核：只在尾端一侧拉出尾巴
        // filter2D: dst(x) = Σ K(i) * src(x + i - anchor)
        cv::Mat kernel = cv::Mat::zeros(1, tailLen + 1, CV_32F);
        kernel.at<float>(0, tailLen) = 1.0f;               // 自身
        for (int i = 0; i < tailLen; ++i)
            kernel.at<float>(0, i) = static_cast<float>(
                0.55 * std::exp(-2.0 * (tailLen - i) / static_cast<double>(tailLen)));

        cv::Point anchor(tailLen, 0);
        if (!toRight)
        {
            cv::Mat flipped;
            cv::flip(kernel, flipped, 1);
            kernel = flipped;
            anchor = cv::Point(0, 0);
        }

        cv::Mat streak;
        cv::filter2D(band, streak, CV_32F, kernel, anchor);
        // max(a,b) = a + max(b-a, 0)，全部支持 CV_32F
        cv::Mat diff;
        cv::subtract(streak, band, diff);       // diff = streak - band
        cv::threshold(diff, diff, 0, 0, cv::THRESH_TOZERO); // diff = max(diff, 0)
        cv::add(band, diff, band);              // band += max(streak-band, 0)
    }

    // ---------------- 收尾：截断并同步回 m_mask / m_alpha_mask ----------------
    a.setTo(0.0f, a < 0.0f);
    a.setTo(1.0f, a > 1.0f);

    a.convertTo(m_alpha_mask, CV_8U, 255.0);
    update_mask_from_alpha(a, m_mask, m_foreground, m_background);
}


//------------------------------------------------------------------------------
// 贴图
//------------------------------------------------------------------------------
cv::Mat DotNumber::paste_mask(cv::Mat bg, cv::Point loc)
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_alpha_mask.empty())
        update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);

    return do_paste(m_mask, m_alpha_mask, bg, loc, color, maskRotate);
}


cv::Mat DotNumber::paste_mask(cv::Mat bg, cv::Point loc, const cv::Scalar& inkColor)
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_alpha_mask.empty())
        update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);

    return do_paste(m_mask, m_alpha_mask, bg, loc, inkColor, maskRotate);
}


//------------------------------------------------------------------------------
// 以固定角度旋转后贴图（批量合成/标注用，同时返回旋转信息）
//------------------------------------------------------------------------------
cv::Mat DotNumber::paste_mask(cv::Mat bg, cv::Point loc, const cv::Scalar& inkColor,
                              double fixedAngleDeg, PasteInfo* info)
{
    if (m_mask.empty())
        gen_mask_mat();
    if (m_alpha_mask.empty())
        update_alpha(m_mask, m_alpha_mask, m_foreground, m_background);

    return do_paste(m_mask, m_alpha_mask, bg, loc, inkColor,
                    static_cast<float>(maskRotate), info, true, fixedAngleDeg);
}


//------------------------------------------------------------------------------
// 标注支持：逐行逐字符计算“最小外接矩形”（mask 坐标系）
// 在字符所属格子(带 2px 容差)内扫描墨迹(alpha>0)得到，故对 1/7 等瘦字形也准确。
//------------------------------------------------------------------------------
std::vector<std::vector<cv::Rect>> DotNumber::get_digit_rects() const
{
    std::vector<std::vector<cv::Rect>> out;
    if (m_DigitLines.empty() || !m_layoutValid)
        return out;
    if (m_mask.empty())
        return out;

    cv::Mat alpha = m_alpha_mask;
    if (alpha.empty())
        update_alpha(m_mask, alpha, m_foreground, m_background);
    if (alpha.empty())
        return out;

    cv::Mat ink;
    cv::threshold(alpha, ink, 0, 255, cv::THRESH_BINARY);   // alpha>0 视为墨迹

    const int M = 2;                       // 格子外容差(容纳漂移/轻微模糊)
    const cv::Rect full(0, 0, ink.cols, ink.rows);
    out.assign(m_DigitLines.size(), std::vector<cv::Rect>());

    for (size_t li = 0; li < m_DigitLines.size(); ++li)
    {
        const std::vector<Digit>& line = m_DigitLines[li];
        const int y0 = m_layoutOffY +
            static_cast<int>(li) * (m_layoutContH + m_layoutLineSpac);
        std::vector<cv::Rect>& rowOut = out[li];
        rowOut.reserve(line.size());

        for (size_t k = 0; k < line.size(); ++k)
        {
            const int x0 = m_layoutOffX +
                static_cast<int>(k) * (m_layoutContW + m_layoutNumSpac);
            const cv::Rect win(x0 - M, y0 - M,
                               m_layoutContW + 2 * M, m_layoutContH + 2 * M);
            const cv::Rect valid = win & full;
            if (valid.width <= 0 || valid.height <= 0)
            {
                rowOut.emplace_back();     // 空框
                continue;
            }

            std::vector<cv::Point> pts;
            cv::findNonZero(ink(valid), pts);
            if (pts.empty())
            {
                rowOut.emplace_back();
                continue;
            }

            int minX = (1 << 30), minY = (1 << 30), maxX = -1, maxY = -1;
            for (const cv::Point& p : pts)
            {
                minX = min(minX, p.x);  maxX = max(maxX, p.x);
                minY = min(minY, p.y);  maxY = max(maxY, p.y);
            }
            rowOut.emplace_back(valid.x + minX, valid.y + minY,
                                maxX - minX + 1, maxY - minY + 1);
        }
    }
    return out;
}
