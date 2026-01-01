//#pragma once
//#ifndef GLCM_EXTRACTOR_H
//#define GLCM_EXTRACTOR_H
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include <vector>
//#include <map>
//#include <string>
//#include <functional>
//
///**
// * @brief 基于灰度共生矩阵的水体/林地提取器（DNN增强版）
// *
// * 功能特点：
// * 1. 传统GLCM纹理特征提取
// * 2. 可选的DNN深度特征增强
// * 3. 多尺度特征融合
// * 4. 自适应阈值分割
// */
//class GLCMExtractor {
//public:
//    /**
//     * @brief 构造函数
//     * @param distances 像素对距离列表
//     * @param angles 像素对角度列表（度数）
//     * @param levels 灰度级数
//     */
//    GLCMExtractor(const std::vector<int>& distances = { 1, 3, 5 },
//        const std::vector<double>& angles = { 0, 45, 90, 135 },
//        int levels = 256);
//
//    /**
//     * @brief 计算单个窗口的GLCM矩阵
//     */
//    cv::Mat computeGLCM(const cv::Mat& window, int distance, double angle);
//
//    /**
//     * @brief 从GLCM矩阵提取纹理特征
//     */
//    std::map<std::string, double> extractTextureFeatures(const cv::Mat& glcm);
//
//    /**
//     * @brief 提取整幅图像的GLCM特征图
//     */
//    std::map<std::string, cv::Mat> extractFeatureMaps(
//        const cv::Mat& image,
//        int windowSize = 15,
//        std::function<void(int)> progress = nullptr);
//
//    /**
//     * @brief 基于GLCM特征分割水体和林地
//     */
//    void segmentWaterForest(
//        const std::map<std::string, cv::Mat>& features,
//        cv::Mat& waterMask,
//        cv::Mat& forestMask,
//        const std::map<std::string, int>& params);
//
//    /**
//     * @brief 处理单张图像的完整流程
//     */
//    std::map<std::string, cv::Mat> processImage(
//        const std::string& imagePath,
//        int windowSize = 15,
//        const std::map<std::string, int>& params = {});
//
//    /**
//     * @brief 可视化结果
//     */
//    void visualizeResults(
//        const std::map<std::string, cv::Mat>& results,
//        const std::string& savePath = "");
//
//    // ================== DNN增强功能 ==================
//
//    /**
//     * @brief 使用DNN提取深度特征
//     * @param image 输入图像
//     * @return 深度特征图
//     */
//    cv::Mat extractDNNFeatures(const cv::Mat& image);
//
//    /**
//     * @brief 加载预训练DNN模型
//     * @param modelPath 模型路径
//     * @param configPath 配置文件路径
//     * @return 是否加载成功
//     */
//    bool loadDNNModel(const std::string& modelPath, const std::string& configPath = "");
//
//    /**
//     * @brief 使用DNN进行语义分割
//     * @param image 输入图像
//     * @return 分割结果
//     */
//    cv::Mat dnnSegmentation(const cv::Mat& image);
//
//    /**
//     * @brief 融合GLCM特征和DNN特征
//     * @param glcmFeatures GLCM特征图
//     * @param dnnFeatures DNN特征图
//     * @param alpha GLCM权重 (0-1)
//     * @return 融合后的特征图
//     */
//    cv::Mat fuseFeatures(const cv::Mat& glcmFeatures,
//        const cv::Mat& dnnFeatures,
//        double alpha = 0.6);
//
//    /**
//     * @brief 使用OpenCV DNN模块进行边缘检测增强
//     * @param image 输入图像
//     * @return 边缘增强图像
//     */
//    cv::Mat dnnEdgeEnhancement(const cv::Mat& image);
//
//    /**
//     * @brief 多尺度特征提取
//     * @param image 输入图像
//     * @param scales 尺度列表
//     * @return 多尺度特征图
//     */
//    std::map<std::string, cv::Mat> multiScaleFeatureExtraction(
//        const cv::Mat& image,
//        const std::vector<double>& scales = { 0.5, 1.0, 2.0 });
//
//    /**
//     * @brief 自适应阈值计算
//     * @param featureMap 特征图
//     * @return 最优阈值
//     */
//    double computeAdaptiveThreshold(const cv::Mat& featureMap);
//
//    /**
//     * @brief DNN增强的完整处理流程
//     * @param imagePath 图像路径
//     * @param windowSize GLCM窗口大小
//     * @param params 分割参数
//     * @param useDNN 是否使用DNN增强
//     * @return 处理结果
//     */
//    std::map<std::string, cv::Mat> processImageEnhanced(
//        const std::string& imagePath,
//        int windowSize = 15,
//        const std::map<std::string, int>& params = {},
//        bool useDNN = true);
//
//    // ================== Gabor滤波器增强 ==================
//
//    /**
//     * @brief 创建Gabor滤波器组
//     * @param ksize 核大小
//     * @param sigma 高斯包络标准差
//     * @param lambd 正弦波波长
//     * @param gamma 空间纵横比
//     * @param psi 相位偏移
//     * @return Gabor滤波器组
//     */
//    std::vector<cv::Mat> createGaborFilterBank(
//        int ksize = 31,
//        double sigma = 4.0,
//        double lambd = 10.0,
//        double gamma = 0.5,
//        double psi = 0);
//
//    /**
//     * @brief 应用Gabor滤波器提取纹理特征
//     * @param image 输入图像
//     * @return Gabor纹理特征图
//     */
//    cv::Mat extractGaborFeatures(const cv::Mat& image);
//
//    // ================== 统计信息 ==================
//
//    /**
//     * @brief 计算分割结果的统计信息
//     */
//    struct SegmentationStats {
//        int waterPixels;
//        int forestPixels;
//        int totalPixels;
//        double waterRatio;
//        double forestRatio;
//        double waterArea;    // 平方米（如果知道分辨率）
//        double forestArea;
//    };
//
//    SegmentationStats computeStats(const cv::Mat& waterMask,
//        const cv::Mat& forestMask,
//        double pixelSize = 1.0);
//
//private:
//    std::vector<int> m_distances;
//    std::vector<double> m_angles;
//    int m_levels;
//
//    // DNN相关成员
//    cv::dnn::Net m_net;
//    bool m_dnnLoaded;
//
//    // Gabor滤波器组
//    std::vector<cv::Mat> m_gaborFilters;
//
//    void normalizeGLCM(cv::Mat& glcm);
//    void morphologicalPostProcess(cv::Mat& mask);
//    double computeContrast(const cv::Mat& glcm);
//    double computeHomogeneity(const cv::Mat& glcm);
//    double computeEnergy(const cv::Mat& glcm);
//    double computeCorrelation(const cv::Mat& glcm);
//    double computeEntropy(const cv::Mat& glcm);
//
//    // 辅助函数
//    cv::Mat applyGaborFilter(const cv::Mat& image, const cv::Mat& kernel);
//    cv::Mat createEdgeDetectionKernel();
//};
//
//#endif // GLCM_EXTRACTOR_H















/**
 * 轻量级GLCM水体/林地提取器 - 优化版
 * 文件名: GLCMextractor.h
 *
 * 优化策略：
 * 1. 双向GLCM（水平+垂直）提高鲁棒性
 * 2. 增加能量特征区分均匀区域
 * 3. 结合局部均值辅助判断
 * 4. 自适应阈值分割
 * 5. 改进形态学后处理
 */

 /**
  * 轻量级GLCM水体/林地提取器 - 优化版
  * 文件名: GLCMextractor.h
  *
  * 优化策略：
  * 1. 双向GLCM（水平+垂直）提高鲁棒性
  * 2. 增加能量特征区分均匀区域
  * 3. 结合局部均值辅助判断
  * 4. 自适应阈值分割
  * 5. 改进形态学后处理
  */

#pragma once
#ifndef GLCM_EXTRACTOR_H
#define GLCM_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

class GLCMExtractor {
public:
    // 处理结果结构
    struct Result {
        cv::Mat original;
        cv::Mat gray;
        cv::Mat waterMask;
        cv::Mat forestMask;
        cv::Mat overlay;
        cv::Mat contrast;
        cv::Mat homogeneity;
        cv::Mat energy;
        cv::Mat meanMap;
        double waterPercent;
        double forestPercent;
        double processTime;
    };

    // 处理参数
    struct Params {
        int grayLevels = 16;        // 增加到16级，提高精度
        int blockSize = 8;          // 减小块大小，提高分辨率
        int minArea = 100;          // 最小区域面积（像素）
        bool useAdaptiveThresh = true;  // 使用自适应阈值
        // 手动阈值（当useAdaptiveThresh=false时使用）
        int waterHomoMin = 180;
        int waterContrastMax = 50;
        int forestHomoMax = 120;
        int forestContrastMin = 80;
    };

    // 构造函数
    GLCMExtractor() {}

    // 主处理函数
    Result process(const std::string& imagePath, const Params& params = Params());
    Result process(const cv::Mat& image, const Params& params = Params());

    // 兼容旧接口
    std::map<std::string, cv::Mat> processImage(
        const std::string& imagePath,
        int windowSize = 15,
        const std::map<std::string, int>& paramsMap = {});

private:
    // 计算单个块的GLCM特征
    void computeBlockFeatures(const cv::Mat& block, int levels,
        float& contrast, float& homogeneity, float& energy, float& mean);

    // 去除小区域
    void removeSmallRegions(cv::Mat& mask, int minArea);
};

// ==================== 实现部分 ====================

inline void GLCMExtractor::computeBlockFeatures(const cv::Mat& block, int levels,
    float& contrast, float& homogeneity, float& energy, float& meanVal) {

    int rows = block.rows;
    int cols = block.cols;

    // GLCM数组
    std::vector<int> glcm(levels * levels, 0);
    int count = 0;
    float sum = 0;

    // 双向统计：水平和垂直
    for (int y = 0; y < rows; y++) {
        const uchar* row = block.ptr<uchar>(y);
        for (int x = 0; x < cols; x++) {
            sum += row[x];

            // 水平方向
            if (x < cols - 1) {
                int g1 = row[x];
                int g2 = row[x + 1];
                glcm[g1 * levels + g2]++;
                glcm[g2 * levels + g1]++;  // 对称
                count += 2;
            }
            // 垂直方向
            if (y < rows - 1) {
                int g1 = row[x];
                int g2 = block.at<uchar>(y + 1, x);
                glcm[g1 * levels + g2]++;
                glcm[g2 * levels + g1]++;  // 对称
                count += 2;
            }
        }
    }

    meanVal = sum / (rows * cols);

    if (count == 0) {
        contrast = 0;
        homogeneity = 1;
        energy = 1;
        return;
    }

    // 计算特征
    float invCount = 1.0f / count;
    contrast = 0;
    homogeneity = 0;
    energy = 0;

    for (int i = 0; i < levels; i++) {
        for (int j = 0; j < levels; j++) {
            int v = glcm[i * levels + j];
            if (v > 0) {
                float p = v * invCount;
                int diff = i - j;
                contrast += p * diff * diff;
                homogeneity += p / (1.0f + std::abs(diff));
                energy += p * p;
            }
        }
    }
}

inline void GLCMExtractor::removeSmallRegions(cv::Mat& mask, int minArea) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < minArea) {
            cv::drawContours(mask, contours, (int)i, cv::Scalar(0), -1);
        }
    }
}

inline GLCMExtractor::Result GLCMExtractor::process(const std::string& imagePath, const Params& params) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + imagePath);
    }
    return process(img, params);
}

inline GLCMExtractor::Result GLCMExtractor::process(const cv::Mat& image, const Params& params) {
    auto startTime = std::chrono::high_resolution_clock::now();

    Result result;
    result.original = image.clone();

    int rows = image.rows;
    int cols = image.cols;
    int levels = params.grayLevels;
    int blockSize = params.blockSize;

    std::cout << "Processing " << cols << "x" << rows << " image..." << std::endl;
    std::cout << "  Block size: " << blockSize << ", Gray levels: " << levels << std::endl;

    // Step 1: 转灰度 + 预处理
    if (image.channels() == 3) {
        cv::cvtColor(image, result.gray, cv::COLOR_BGR2GRAY);
    }
    else {
        result.gray = image.clone();
    }

    // 轻度高斯模糊减少噪声
    cv::Mat smoothed;
    cv::GaussianBlur(result.gray, smoothed, cv::Size(3, 3), 0);

    // Step 2: 量化到指定灰度级
    cv::Mat quantized(smoothed.size(), CV_8U);
    double scale = (double)levels / 256.0;
    for (int i = 0; i < rows; i++) {
        const uchar* src = smoothed.ptr<uchar>(i);
        uchar* dst = quantized.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            int v = (int)(src[j] * scale);
            dst[j] = (v >= levels) ? (levels - 1) : v;
        }
    }

    // Step 3: 分块计算GLCM特征
    int blockRows = (rows + blockSize - 1) / blockSize;
    int blockCols = (cols + blockSize - 1) / blockSize;

    cv::Mat contrastMap(blockRows, blockCols, CV_32F, cv::Scalar(0));
    cv::Mat homogeneityMap(blockRows, blockCols, CV_32F, cv::Scalar(0));
    cv::Mat energyMap(blockRows, blockCols, CV_32F, cv::Scalar(0));
    cv::Mat meanMap(blockRows, blockCols, CV_32F, cv::Scalar(0));

    for (int bi = 0; bi < blockRows; bi++) {
        int y0 = bi * blockSize;
        int y1 = std::min(y0 + blockSize, rows);

        for (int bj = 0; bj < blockCols; bj++) {
            int x0 = bj * blockSize;
            int x1 = std::min(x0 + blockSize, cols);

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            cv::Mat block = quantized(roi);

            float contrast, homogeneity, energy, mean;
            computeBlockFeatures(block, levels, contrast, homogeneity, energy, mean);

            contrastMap.at<float>(bi, bj) = contrast;
            homogeneityMap.at<float>(bi, bj) = homogeneity;
            energyMap.at<float>(bi, bj) = energy;
            meanMap.at<float>(bi, bj) = mean;
        }
    }

    // Step 4: 上采样到原图大小（使用双三次插值更平滑）
    cv::resize(contrastMap, result.contrast, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
    cv::resize(homogeneityMap, result.homogeneity, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
    cv::resize(energyMap, result.energy, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);
    cv::resize(meanMap, result.meanMap, cv::Size(cols, rows), 0, 0, cv::INTER_CUBIC);

    // 归一化到0-255
    cv::Mat contrastNorm, homogeneityNorm, energyNorm;
    cv::normalize(result.contrast, contrastNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(result.homogeneity, homogeneityNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(result.energy, energyNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Step 5: 分割
    cv::Mat waterMask, forestMask;

    if (params.useAdaptiveThresh) {
        // 自适应阈值分割
        // 水体特征：高同质性 + 低对比度 + 高能量（均匀区域）
        cv::Mat waterScore;
        cv::addWeighted(homogeneityNorm, 0.5, energyNorm, 0.3, 0, waterScore);
        cv::Mat contrastInv;
        cv::subtract(cv::Scalar(255), contrastNorm, contrastInv);
        cv::addWeighted(waterScore, 1.0, contrastInv, 0.2, 0, waterScore);

        // Otsu自动阈值
        cv::threshold(waterScore, waterMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 林地特征：低同质性 + 高对比度（纹理丰富）
        cv::Mat forestScore;
        cv::Mat homoInv;
        cv::subtract(cv::Scalar(255), homogeneityNorm, homoInv);
        cv::addWeighted(contrastNorm, 0.6, homoInv, 0.4, 0, forestScore);

        cv::threshold(forestScore, forestMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 排除水体区域
        cv::Mat notWater;
        cv::bitwise_not(waterMask, notWater);
        cv::bitwise_and(forestMask, notWater, forestMask);
    }
    else {
        // 手动阈值
        waterMask = (homogeneityNorm > params.waterHomoMin) &
            (contrastNorm < params.waterContrastMax);
        forestMask = (homogeneityNorm < params.forestHomoMax) &
            (contrastNorm > params.forestContrastMin);
    }

    // Step 6: 形态学后处理
    cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernelMedium = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernelLarge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    // 水体：先闭合填充孔洞，再开运算去噪
    cv::morphologyEx(waterMask, waterMask, cv::MORPH_CLOSE, kernelLarge);
    cv::morphologyEx(waterMask, waterMask, cv::MORPH_OPEN, kernelMedium);

    // 林地：类似处理
    cv::morphologyEx(forestMask, forestMask, cv::MORPH_CLOSE, kernelMedium);
    cv::morphologyEx(forestMask, forestMask, cv::MORPH_OPEN, kernelSmall);

    // 去除小区域
    removeSmallRegions(waterMask, params.minArea);
    removeSmallRegions(forestMask, params.minArea);

    result.waterMask = waterMask;
    result.forestMask = forestMask;

    // Step 7: 生成叠加可视化
    result.overlay = image.clone();

    // 创建彩色叠加层
    cv::Mat waterOverlay = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat forestOverlay = cv::Mat::zeros(image.size(), CV_8UC3);

    waterOverlay.setTo(cv::Scalar(255, 150, 0), waterMask);   // 蓝色水体
    forestOverlay.setTo(cv::Scalar(0, 200, 50), forestMask);  // 绿色林地

    // 半透明叠加
    cv::addWeighted(result.overlay, 0.6, waterOverlay, 0.4, 0, result.overlay);
    cv::addWeighted(result.overlay, 1.0, forestOverlay, 0.4, 0, result.overlay);

    // 绘制边界线使结果更清晰
    std::vector<std::vector<cv::Point>> waterContours, forestContours;
    cv::findContours(waterMask.clone(), waterContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(forestMask.clone(), forestContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::drawContours(result.overlay, waterContours, -1, cv::Scalar(255, 200, 50), 2);
    cv::drawContours(result.overlay, forestContours, -1, cv::Scalar(50, 255, 100), 2);

    // 统计
    int totalPixels = rows * cols;
    result.waterPercent = 100.0 * cv::countNonZero(result.waterMask) / totalPixels;
    result.forestPercent = 100.0 * cv::countNonZero(result.forestMask) / totalPixels;

    auto endTime = std::chrono::high_resolution_clock::now();
    result.processTime = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << "Done in " << std::fixed << std::setprecision(2)
        << result.processTime << "s" << std::endl;
    std::cout << "  Water:  " << std::setprecision(1) << result.waterPercent << "%" << std::endl;
    std::cout << "  Forest: " << std::setprecision(1) << result.forestPercent << "%" << std::endl;

    return result;
}

// 兼容旧接口
inline std::map<std::string, cv::Mat> GLCMExtractor::processImage(
    const std::string& imagePath,
    int windowSize,
    const std::map<std::string, int>& paramsMap) {

    Params params;
    if (paramsMap.count("water_homo")) params.waterHomoMin = paramsMap.at("water_homo");
    if (paramsMap.count("water_contrast")) params.waterContrastMax = paramsMap.at("water_contrast");
    if (paramsMap.count("forest_homo")) params.forestHomoMax = paramsMap.at("forest_homo");
    if (paramsMap.count("forest_contrast")) params.forestContrastMin = paramsMap.at("forest_contrast");
    if (paramsMap.count("block_size")) params.blockSize = paramsMap.at("block_size");
    if (paramsMap.count("adaptive")) params.useAdaptiveThresh = (paramsMap.at("adaptive") != 0);

    Result r = process(imagePath, params);

    std::map<std::string, cv::Mat> results;
    results["original"] = r.original;
    results["gray"] = r.gray;
    results["contrast"] = r.contrast;
    results["homogeneity"] = r.homogeneity;
    results["energy"] = r.energy;
    results["water_mask"] = r.waterMask;
    results["forest_mask"] = r.forestMask;
    results["overlay"] = r.overlay;

    return results;
}

#endif // GLCM_EXTRACTOR_H