//#include "GLCMExtractor.h"
//#include <cmath>
//#include <iostream>
//#include <algorithm>
//#include <numeric>
//
//GLCMExtractor::GLCMExtractor(const std::vector<int>& distances,
//    const std::vector<double>& angles,
//    int levels)
//    : m_distances(distances), m_angles(angles), m_levels(levels), m_dnnLoaded(false) {
//    // 初始化Gabor滤波器组
//    m_gaborFilters = createGaborFilterBank();
//}
//
//// ================== 基础GLCM功能 ==================
//
//cv::Mat GLCMExtractor::computeGLCM(const cv::Mat& window, int distance, double angle) {
//    cv::Mat glcm = cv::Mat::zeros(m_levels, m_levels, CV_64F);
//
//    double rad = angle * CV_PI / 180.0;
//    int dx = static_cast<int>(std::round(distance * std::cos(rad)));
//    int dy = static_cast<int>(std::round(distance * std::sin(rad)));
//
//    int rows = window.rows;
//    int cols = window.cols;
//
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            int ni = i + dy;
//            int nj = j + dx;
//
//            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
//                int gray1 = window.at<uchar>(i, j);
//                int gray2 = window.at<uchar>(ni, nj);
//
//                // 量化到指定级数
//                gray1 = gray1 * m_levels / 256;
//                gray2 = gray2 * m_levels / 256;
//
//                if (gray1 >= m_levels) gray1 = m_levels - 1;
//                if (gray2 >= m_levels) gray2 = m_levels - 1;
//
//                glcm.at<double>(gray1, gray2) += 1.0;
//                glcm.at<double>(gray2, gray1) += 1.0;
//            }
//        }
//    }
//
//    normalizeGLCM(glcm);
//    return glcm;
//}
//
//void GLCMExtractor::normalizeGLCM(cv::Mat& glcm) {
//    double sum = cv::sum(glcm)[0];
//    if (sum > 0) {
//        glcm /= sum;
//    }
//}
//
//double GLCMExtractor::computeContrast(const cv::Mat& glcm) {
//    double contrast = 0.0;
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            contrast += glcm.at<double>(i, j) * std::pow(i - j, 2);
//        }
//    }
//    return contrast;
//}
//
//double GLCMExtractor::computeHomogeneity(const cv::Mat& glcm) {
//    double homogeneity = 0.0;
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            homogeneity += glcm.at<double>(i, j) / (1.0 + std::abs(i - j));
//        }
//    }
//    return homogeneity;
//}
//
//double GLCMExtractor::computeEnergy(const cv::Mat& glcm) {
//    double energy = 0.0;
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            energy += std::pow(glcm.at<double>(i, j), 2);
//        }
//    }
//    return energy;
//}
//
//double GLCMExtractor::computeCorrelation(const cv::Mat& glcm) {
//    double meanI = 0.0, meanJ = 0.0;
//    double stdI = 0.0, stdJ = 0.0;
//
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            meanI += i * glcm.at<double>(i, j);
//            meanJ += j * glcm.at<double>(i, j);
//        }
//    }
//
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            stdI += std::pow(i - meanI, 2) * glcm.at<double>(i, j);
//            stdJ += std::pow(j - meanJ, 2) * glcm.at<double>(i, j);
//        }
//    }
//
//    stdI = std::sqrt(stdI);
//    stdJ = std::sqrt(stdJ);
//
//    if (stdI < 1e-10 || stdJ < 1e-10) return 0.0;
//
//    double correlation = 0.0;
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            correlation += (i - meanI) * (j - meanJ) * glcm.at<double>(i, j);
//        }
//    }
//
//    return correlation / (stdI * stdJ);
//}
//
//double GLCMExtractor::computeEntropy(const cv::Mat& glcm) {
//    double entropy = 0.0;
//    for (int i = 0; i < glcm.rows; i++) {
//        for (int j = 0; j < glcm.cols; j++) {
//            double p = glcm.at<double>(i, j);
//            if (p > 1e-10) {
//                entropy -= p * std::log2(p);
//            }
//        }
//    }
//    return entropy;
//}
//
//std::map<std::string, double> GLCMExtractor::extractTextureFeatures(const cv::Mat& glcm) {
//    std::map<std::string, double> features;
//    features["contrast"] = computeContrast(glcm);
//    features["homogeneity"] = computeHomogeneity(glcm);
//    features["energy"] = computeEnergy(glcm);
//    features["correlation"] = computeCorrelation(glcm);
//    features["entropy"] = computeEntropy(glcm);
//    return features;
//}
//
//std::map<std::string, cv::Mat> GLCMExtractor::extractFeatureMaps(
//    const cv::Mat& image,
//    int windowSize,
//    std::function<void(int)> progress) {
//
//    std::cout << "开始提取GLCM特征..." << std::endl;
//
//    int rows = image.rows;
//    int cols = image.cols;
//    int pad = windowSize / 2;
//
//    cv::Mat contrastMap = cv::Mat::zeros(rows, cols, CV_64F);
//    cv::Mat homogeneityMap = cv::Mat::zeros(rows, cols, CV_64F);
//    cv::Mat energyMap = cv::Mat::zeros(rows, cols, CV_64F);
//    cv::Mat correlationMap = cv::Mat::zeros(rows, cols, CV_64F);
//    cv::Mat entropyMap = cv::Mat::zeros(rows, cols, CV_64F);
//
//    cv::Mat padded;
//    cv::copyMakeBorder(image, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);
//
//    int step = 2;
//    int totalIterations = (rows / step) * (cols / step);
//    int currentIteration = 0;
//
//#pragma omp parallel for collapse(2) schedule(dynamic)
//    for (int i = 0; i < rows; i += step) {
//        for (int j = 0; j < cols; j += step) {
//            cv::Rect roi(j, i, windowSize, windowSize);
//            cv::Mat window = padded(roi);
//
//            double contrast = 0.0, homogeneity = 0.0, energy = 0.0;
//            double correlation = 0.0, entropy = 0.0;
//            int count = 0;
//
//            for (int d : m_distances) {
//                for (double angle : m_angles) {
//                    cv::Mat glcm = computeGLCM(window, d, angle);
//                    auto features = extractTextureFeatures(glcm);
//
//                    contrast += features["contrast"];
//                    homogeneity += features["homogeneity"];
//                    energy += features["energy"];
//                    correlation += features["correlation"];
//                    entropy += features["entropy"];
//                    count++;
//                }
//            }
//
//            for (int di = 0; di < step && i + di < rows; di++) {
//                for (int dj = 0; dj < step && j + dj < cols; dj++) {
//                    contrastMap.at<double>(i + di, j + dj) = contrast / count;
//                    homogeneityMap.at<double>(i + di, j + dj) = homogeneity / count;
//                    energyMap.at<double>(i + di, j + dj) = energy / count;
//                    correlationMap.at<double>(i + di, j + dj) = correlation / count;
//                    entropyMap.at<double>(i + di, j + dj) = entropy / count;
//                }
//            }
//
//#pragma omp critical
//            {
//                currentIteration++;
//                if (progress && currentIteration % 1000 == 0) {
//                    progress(currentIteration * 100 / totalIterations);
//                }
//            }
//        }
//    }
//
//    std::cout << "GLCM特征提取完成" << std::endl;
//
//    std::map<std::string, cv::Mat> featureMaps;
//    featureMaps["contrast"] = contrastMap;
//    featureMaps["homogeneity"] = homogeneityMap;
//    featureMaps["energy"] = energyMap;
//    featureMaps["correlation"] = correlationMap;
//    featureMaps["entropy"] = entropyMap;
//
//    return featureMaps;
//}
//
//void GLCMExtractor::morphologicalPostProcess(cv::Mat& mask) {
//    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);
//    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
//
//    // 额外的去噪处理
//    cv::Mat largeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
//    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, largeElement);
//}
//
//void GLCMExtractor::segmentWaterForest(
//    const std::map<std::string, cv::Mat>& features,
//    cv::Mat& waterMask,
//    cv::Mat& forestMask,
//    const std::map<std::string, int>& params) {
//
//    int waterHomoThresh = params.count("water_homo") ? params.at("water_homo") : 200;
//    int waterContrastThresh = params.count("water_contrast") ? params.at("water_contrast") : 50;
//    int forestHomoThresh = params.count("forest_homo") ? params.at("forest_homo") : 150;
//    int forestContrastThresh = params.count("forest_contrast") ? params.at("forest_contrast") : 80;
//
//    cv::Mat homogeneity, contrast, energy, entropy;
//    cv::normalize(features.at("homogeneity"), homogeneity, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(features.at("contrast"), contrast, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(features.at("energy"), energy, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(features.at("entropy"), entropy, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    // 水体分割：高同质性、低对比度、高能量
//    waterMask = (homogeneity > waterHomoThresh) & (contrast < waterContrastThresh);
//
//    // 林地分割：低同质性、高对比度、高熵
//    forestMask = (homogeneity < forestHomoThresh) & (contrast > forestContrastThresh);
//
//    morphologicalPostProcess(waterMask);
//    morphologicalPostProcess(forestMask);
//
//    std::cout << "分割完成" << std::endl;
//}
//
//std::map<std::string, cv::Mat> GLCMExtractor::processImage(
//    const std::string& imagePath,
//    int windowSize,
//    const std::map<std::string, int>& params) {
//
//    std::cout << "\n处理图像: " << imagePath << std::endl;
//
//    cv::Mat imgBGR = cv::imread(imagePath);
//    if (imgBGR.empty()) {
//        throw std::runtime_error("无法读取图像: " + imagePath);
//    }
//
//    cv::Mat imgGray;
//    cv::cvtColor(imgBGR, imgGray, cv::COLOR_BGR2GRAY);
//
//    auto features = extractFeatureMaps(imgGray, windowSize);
//
//    cv::Mat waterMask, forestMask;
//    segmentWaterForest(features, waterMask, forestMask, params);
//
//    cv::Mat resultImg = imgBGR.clone();
//    resultImg.setTo(cv::Scalar(255, 0, 0), waterMask);
//    resultImg.setTo(cv::Scalar(0, 255, 0), forestMask);
//
//    cv::Mat overlay;
//    cv::addWeighted(imgBGR, 0.6, resultImg, 0.4, 0, overlay);
//
//    auto stats = computeStats(waterMask, forestMask);
//    std::cout << "水体像素数: " << stats.waterPixels << " (" << stats.waterRatio * 100 << "%)" << std::endl;
//    std::cout << "林地像素数: " << stats.forestPixels << " (" << stats.forestRatio * 100 << "%)" << std::endl;
//
//    std::map<std::string, cv::Mat> results;
//    results["original"] = imgBGR;
//    results["gray"] = imgGray;
//    results["contrast"] = features["contrast"];
//    results["homogeneity"] = features["homogeneity"];
//    results["energy"] = features["energy"];
//    results["correlation"] = features["correlation"];
//    results["entropy"] = features["entropy"];
//    results["water_mask"] = waterMask;
//    results["forest_mask"] = forestMask;
//    results["result"] = resultImg;
//    results["overlay"] = overlay;
//
//    return results;
//}
//
//// ================== DNN增强功能 ==================
//
//bool GLCMExtractor::loadDNNModel(const std::string& modelPath, const std::string& configPath) {
//    try {
//        if (configPath.empty()) {
//            m_net = cv::dnn::readNet(modelPath);
//        }
//        else {
//            m_net = cv::dnn::readNet(modelPath, configPath);
//        }
//        m_dnnLoaded = !m_net.empty();
//
//        if (m_dnnLoaded) {
//            // 设置后端和目标
//            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//            std::cout << "DNN模型加载成功" << std::endl;
//        }
//
//        return m_dnnLoaded;
//    }
//    catch (const cv::Exception& e) {
//        std::cerr << "加载DNN模型失败: " << e.what() << std::endl;
//        m_dnnLoaded = false;
//        return false;
//    }
//}
//
//cv::Mat GLCMExtractor::extractDNNFeatures(const cv::Mat& image) {
//    cv::Mat result;
//
//    if (!m_dnnLoaded) {
//        // 如果没有加载DNN模型，使用OpenCV内置的边缘检测作为替代
//        cv::Mat gray;
//        if (image.channels() == 3) {
//            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//        }
//        else {
//            gray = image.clone();
//        }
//
//        // 使用Canny边缘检测 + Laplacian 作为简单的"特征"
//        cv::Mat edges, laplacian;
//        cv::Canny(gray, edges, 50, 150);
//        cv::Laplacian(gray, laplacian, CV_64F);
//        cv::convertScaleAbs(laplacian, laplacian);
//
//        // 合并边缘特征
//        cv::addWeighted(edges, 0.5, laplacian, 0.5, 0, result);
//        return result;
//    }
//
//    // 使用DNN模型提取特征
//    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(224, 224),
//        cv::Scalar(0, 0, 0), true, false);
//    m_net.setInput(blob);
//    cv::Mat output = m_net.forward();
//
//    // 处理输出
//    cv::resize(output, result, image.size());
//
//    return result;
//}
//
//cv::Mat GLCMExtractor::dnnEdgeEnhancement(const cv::Mat& image) {
//    cv::Mat gray;
//    if (image.channels() == 3) {
//        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        gray = image.clone();
//    }
//
//    // 使用OpenCV DNN创建自定义卷积核进行边缘检测
//    // Sobel-like kernels
//    cv::Mat sobelX, sobelY;
//    cv::Sobel(gray, sobelX, CV_64F, 1, 0, 3);
//    cv::Sobel(gray, sobelY, CV_64F, 0, 1, 3);
//
//    cv::Mat magnitude;
//    cv::magnitude(sobelX, sobelY, magnitude);
//    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    // 使用DNN风格的多尺度边缘检测
//    std::vector<cv::Mat> multiScaleEdges;
//    for (int ksize : {3, 5, 7}) {
//        cv::Mat edgeK;
//        cv::Canny(gray, edgeK, 30, 100, ksize);
//        multiScaleEdges.push_back(edgeK);
//    }
//
//    // 融合多尺度边缘
//    cv::Mat fusedEdges = cv::Mat::zeros(gray.size(), CV_8U);
//    for (const auto& edge : multiScaleEdges) {
//        cv::add(fusedEdges, edge / (int)multiScaleEdges.size(), fusedEdges);
//    }
//
//    // 与Sobel边缘融合
//    cv::Mat result;
//    cv::addWeighted(magnitude, 0.5, fusedEdges, 0.5, 0, result);
//
//    return result;
//}
//
//cv::Mat GLCMExtractor::dnnSegmentation(const cv::Mat& image) {
//    cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);
//
//    if (!m_dnnLoaded) {
//        // 使用简单的颜色分割作为替代
//        cv::Mat hsv;
//        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
//
//        // 水体通常是蓝色或深色
//        cv::Mat waterMask;
//        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), waterMask);
//
//        // 林地通常是绿色
//        cv::Mat forestMask;
//        cv::inRange(hsv, cv::Scalar(35, 50, 50), cv::Scalar(85, 255, 255), forestMask);
//
//        result.setTo(1, waterMask);
//        result.setTo(2, forestMask);
//
//        return result;
//    }
//
//    // 使用DNN模型进行分割
//    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(512, 512),
//        cv::Scalar(0, 0, 0), true, false);
//    m_net.setInput(blob);
//    cv::Mat output = m_net.forward();
//
//    cv::resize(output, result, image.size());
//
//    return result;
//}
//
//cv::Mat GLCMExtractor::fuseFeatures(const cv::Mat& glcmFeatures,
//    const cv::Mat& dnnFeatures,
//    double alpha) {
//    cv::Mat glcmNorm, dnnNorm;
//    cv::normalize(glcmFeatures, glcmNorm, 0, 1, cv::NORM_MINMAX, CV_64F);
//    cv::normalize(dnnFeatures, dnnNorm, 0, 1, cv::NORM_MINMAX, CV_64F);
//
//    // 调整大小以匹配
//    if (glcmNorm.size() != dnnNorm.size()) {
//        cv::resize(dnnNorm, dnnNorm, glcmNorm.size());
//    }
//
//    cv::Mat fused;
//    cv::addWeighted(glcmNorm, alpha, dnnNorm, 1.0 - alpha, 0, fused);
//
//    return fused;
//}
//
//std::map<std::string, cv::Mat> GLCMExtractor::multiScaleFeatureExtraction(
//    const cv::Mat& image,
//    const std::vector<double>& scales) {
//
//    std::map<std::string, cv::Mat> multiScaleFeatures;
//    std::map<std::string, cv::Mat> accumFeatures;
//
//    cv::Mat gray;
//    if (image.channels() == 3) {
//        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        gray = image.clone();
//    }
//
//    for (double scale : scales) {
//        cv::Mat scaled;
//        cv::resize(gray, scaled, cv::Size(), scale, scale);
//
//        // 在每个尺度提取GLCM特征
//        int windowSize = static_cast<int>(15 * scale);
//        if (windowSize % 2 == 0) windowSize++;
//        if (windowSize < 3) windowSize = 3;
//
//        auto features = extractFeatureMaps(scaled, windowSize);
//
//        // 将特征调整回原始大小并累加
//        for (auto& pair : features) {
//            cv::Mat resized;
//            cv::resize(pair.second, resized, gray.size());
//
//            if (accumFeatures.find(pair.first) == accumFeatures.end()) {
//                accumFeatures[pair.first] = cv::Mat::zeros(gray.size(), CV_64F);
//            }
//            accumFeatures[pair.first] += resized / (double)scales.size();
//        }
//    }
//
//    return accumFeatures;
//}
//
//double GLCMExtractor::computeAdaptiveThreshold(const cv::Mat& featureMap) {
//    cv::Mat normalized;
//    cv::normalize(featureMap, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    // 使用Otsu方法计算最优阈值
//    double otsuThresh = cv::threshold(normalized, cv::Mat(), 0, 255,
//        cv::THRESH_BINARY | cv::THRESH_OTSU);
//
//    return otsuThresh;
//}
//
//std::map<std::string, cv::Mat> GLCMExtractor::processImageEnhanced(
//    const std::string& imagePath,
//    int windowSize,
//    const std::map<std::string, int>& params,
//    bool useDNN) {
//
//    std::cout << "\n处理图像(增强模式): " << imagePath << std::endl;
//
//    cv::Mat imgBGR = cv::imread(imagePath);
//    if (imgBGR.empty()) {
//        throw std::runtime_error("无法读取图像: " + imagePath);
//    }
//
//    cv::Mat imgGray;
//    cv::cvtColor(imgBGR, imgGray, cv::COLOR_BGR2GRAY);
//
//    // 提取GLCM特征
//    auto glcmFeatures = extractFeatureMaps(imgGray, windowSize);
//
//    // 提取Gabor特征
//    std::cout << "提取Gabor纹理特征..." << std::endl;
//    cv::Mat gaborFeatures = extractGaborFeatures(imgGray);
//
//    // DNN边缘增强
//    cv::Mat dnnEdges;
//    if (useDNN) {
//        std::cout << "DNN边缘增强处理..." << std::endl;
//        dnnEdges = dnnEdgeEnhancement(imgBGR);
//    }
//
//    // 融合特征
//    cv::Mat fusedHomogeneity = glcmFeatures["homogeneity"].clone();
//    cv::Mat fusedContrast = glcmFeatures["contrast"].clone();
//
//    if (useDNN && !dnnEdges.empty()) {
//        cv::Mat dnnEdgesFloat;
//        dnnEdges.convertTo(dnnEdgesFloat, CV_64F);
//        cv::resize(dnnEdgesFloat, dnnEdgesFloat, fusedContrast.size());
//
//        // 使用边缘信息增强对比度特征
//        fusedContrast = fuseFeatures(fusedContrast, dnnEdgesFloat, 0.7);
//    }
//
//    // 使用Gabor特征增强
//    if (!gaborFeatures.empty()) {
//        cv::Mat gaborFloat;
//        gaborFeatures.convertTo(gaborFloat, CV_64F);
//        cv::resize(gaborFloat, gaborFloat, fusedContrast.size());
//        fusedContrast = fuseFeatures(fusedContrast, gaborFloat, 0.8);
//    }
//
//    // 更新特征图
//    glcmFeatures["homogeneity"] = fusedHomogeneity;
//    glcmFeatures["contrast"] = fusedContrast;
//
//    // 计算自适应阈值
//    double adaptiveWaterHomo = computeAdaptiveThreshold(fusedHomogeneity);
//    double adaptiveForestContrast = computeAdaptiveThreshold(fusedContrast);
//
//    std::cout << "自适应阈值 - 水体同质性: " << adaptiveWaterHomo
//        << ", 林地对比度: " << adaptiveForestContrast << std::endl;
//
//    // 分割
//    cv::Mat waterMask, forestMask;
//    segmentWaterForest(glcmFeatures, waterMask, forestMask, params);
//
//    // 创建结果图像
//    cv::Mat resultImg = imgBGR.clone();
//    resultImg.setTo(cv::Scalar(255, 0, 0), waterMask);
//    resultImg.setTo(cv::Scalar(0, 255, 0), forestMask);
//
//    cv::Mat overlay;
//    cv::addWeighted(imgBGR, 0.6, resultImg, 0.4, 0, overlay);
//
//    auto stats = computeStats(waterMask, forestMask, 2.0); // 假设2米分辨率
//    std::cout << "水体: " << stats.waterPixels << "像素, 面积: "
//        << stats.waterArea / 10000 << " 公顷" << std::endl;
//    std::cout << "林地: " << stats.forestPixels << "像素, 面积: "
//        << stats.forestArea / 10000 << " 公顷" << std::endl;
//
//    std::map<std::string, cv::Mat> results;
//    results["original"] = imgBGR;
//    results["gray"] = imgGray;
//    results["contrast"] = fusedContrast;
//    results["homogeneity"] = fusedHomogeneity;
//    results["energy"] = glcmFeatures["energy"];
//    results["correlation"] = glcmFeatures["correlation"];
//    results["entropy"] = glcmFeatures["entropy"];
//    results["gabor"] = gaborFeatures;
//    results["water_mask"] = waterMask;
//    results["forest_mask"] = forestMask;
//    results["result"] = resultImg;
//    results["overlay"] = overlay;
//
//    if (!dnnEdges.empty()) {
//        results["dnn_edges"] = dnnEdges;
//    }
//
//    return results;
//}
//
//// ================== Gabor滤波器 ==================
//
//std::vector<cv::Mat> GLCMExtractor::createGaborFilterBank(
//    int ksize, double sigma, double lambd, double gamma, double psi) {
//
//    std::vector<cv::Mat> filters;
//    std::vector<double> thetas = { 0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4 };
//    std::vector<double> lambdas = { 8, 12, 16 };
//
//    for (double theta : thetas) {
//        for (double lam : lambdas) {
//            cv::Mat kernel = cv::getGaborKernel(
//                cv::Size(ksize, ksize), sigma, theta, lam, gamma, psi, CV_64F);
//            filters.push_back(kernel);
//        }
//    }
//
//    return filters;
//}
//
//cv::Mat GLCMExtractor::applyGaborFilter(const cv::Mat& image, const cv::Mat& kernel) {
//    cv::Mat filtered;
//    cv::filter2D(image, filtered, CV_64F, kernel);
//    return filtered;
//}
//
//cv::Mat GLCMExtractor::extractGaborFeatures(const cv::Mat& image) {
//    cv::Mat gray;
//    if (image.channels() == 3) {
//        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        gray = image.clone();
//    }
//
//    cv::Mat grayFloat;
//    gray.convertTo(grayFloat, CV_64F);
//
//    cv::Mat response = cv::Mat::zeros(gray.size(), CV_64F);
//
//    for (const auto& kernel : m_gaborFilters) {
//        cv::Mat filtered = applyGaborFilter(grayFloat, kernel);
//        cv::Mat absFiltered;
//        cv::pow(filtered, 2, absFiltered);
//        response += absFiltered;
//    }
//
//    // 归一化
//    cv::sqrt(response / (double)m_gaborFilters.size(), response);
//
//    cv::Mat result;
//    cv::normalize(response, result, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    return result;
//}
//
//// ================== 统计信息 ==================
//
//GLCMExtractor::SegmentationStats GLCMExtractor::computeStats(
//    const cv::Mat& waterMask,
//    const cv::Mat& forestMask,
//    double pixelSize) {
//
//    SegmentationStats stats;
//    stats.waterPixels = cv::countNonZero(waterMask);
//    stats.forestPixels = cv::countNonZero(forestMask);
//    stats.totalPixels = waterMask.rows * waterMask.cols;
//    stats.waterRatio = (double)stats.waterPixels / stats.totalPixels;
//    stats.forestRatio = (double)stats.forestPixels / stats.totalPixels;
//    stats.waterArea = stats.waterPixels * pixelSize * pixelSize;
//    stats.forestArea = stats.forestPixels * pixelSize * pixelSize;
//
//    return stats;
//}
//
//// ================== 可视化 ==================
//
//void GLCMExtractor::visualizeResults(
//    const std::map<std::string, cv::Mat>& results,
//    const std::string& savePath) {
//
//    cv::namedWindow("原始图像", cv::WINDOW_NORMAL);
//    cv::namedWindow("灰度图像", cv::WINDOW_NORMAL);
//    cv::namedWindow("对比度特征", cv::WINDOW_NORMAL);
//    cv::namedWindow("同质性特征", cv::WINDOW_NORMAL);
//    cv::namedWindow("水体提取", cv::WINDOW_NORMAL);
//    cv::namedWindow("林地提取", cv::WINDOW_NORMAL);
//    cv::namedWindow("叠加结果", cv::WINDOW_NORMAL);
//
//    cv::imshow("原始图像", results.at("original"));
//    cv::imshow("灰度图像", results.at("gray"));
//
//    cv::Mat contrastVis, homogeneityVis;
//    cv::normalize(results.at("contrast"), contrastVis, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::normalize(results.at("homogeneity"), homogeneityVis, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    cv::Mat contrastColor, homogeneityColor;
//    cv::applyColorMap(contrastVis, contrastColor, cv::COLORMAP_JET);
//    cv::applyColorMap(homogeneityVis, homogeneityColor, cv::COLORMAP_JET);
//
//    cv::imshow("对比度特征", contrastColor);
//    cv::imshow("同质性特征", homogeneityColor);
//
//    cv::Mat waterVis, forestVis;
//    results.at("water_mask").convertTo(waterVis, CV_8U, 255);
//    results.at("forest_mask").convertTo(forestVis, CV_8U, 255);
//
//    cv::Mat waterColor, forestColor;
//    cv::applyColorMap(waterVis, waterColor, cv::COLORMAP_WINTER);
//    cv::applyColorMap(forestVis, forestColor, cv::COLORMAP_SUMMER);
//
//    cv::imshow("水体提取", waterColor);
//    cv::imshow("林地提取", forestColor);
//    cv::imshow("叠加结果", results.at("overlay"));
//
//    if (!savePath.empty()) {
//        std::string base = savePath.substr(0, savePath.find_last_of("."));
//        cv::imwrite(base + "_contrast.png", contrastColor);
//        cv::imwrite(base + "_homogeneity.png", homogeneityColor);
//        cv::imwrite(base + "_water.png", waterVis);
//        cv::imwrite(base + "_forest.png", forestVis);
//        cv::imwrite(base + "_overlay.png", results.at("overlay"));
//        std::cout << "结果已保存到: " << base << "_*.png" << std::endl;
//    }
//
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//}












/**
 * GLCM算法实现
 * 文件名: GLCM_Extracting.cpp
 *
 * 注意：算法实现已经放在GLCMextractor.h中(header-only)
 * 该文件是为了保持项目结构
 */

#include "GLCMextractor.h"

 // 算法实现已在头文件中，此文件保留为空