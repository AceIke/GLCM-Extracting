//#include "GUIInterface.h"
//#include <iostream>
//#include <sstream>
//#include <iomanip>
//#include <chrono>
//
//// 静态成员初始化
//const std::string GUIInterface::MAIN_WINDOW = "GLCM水体/林地提取 - 主界面";
//const std::string GUIInterface::CONTROL_WINDOW = "参数控制面板";
//const std::string GUIInterface::RESULT_WINDOW = "分割结果";
//const std::string GUIInterface::FEATURE_WINDOW = "特征图可视化";
//
//GUIInterface::GUIInterface()
//    : m_windowSize(15),
//    m_waterHomoThresh(200),
//    m_waterContrastThresh(50),
//    m_forestHomoThresh(150),
//    m_forestContrastThresh(80),
//    m_useDNN(1),
//    m_displayMode(0) {
//}
//
//GUIInterface::~GUIInterface() {
//    cv::destroyAllWindows();
//}
//
//void GUIInterface::createWindows() {
//    // 创建主窗口
//    cv::namedWindow(MAIN_WINDOW, cv::WINDOW_NORMAL);
//    cv::resizeWindow(MAIN_WINDOW, 800, 600);
//
//    // 创建控制面板窗口
//    cv::namedWindow(CONTROL_WINDOW, cv::WINDOW_AUTOSIZE);
//
//    // 创建结果窗口
//    cv::namedWindow(RESULT_WINDOW, cv::WINDOW_NORMAL);
//    cv::resizeWindow(RESULT_WINDOW, 800, 600);
//
//    // 创建特征窗口
//    cv::namedWindow(FEATURE_WINDOW, cv::WINDOW_NORMAL);
//    cv::resizeWindow(FEATURE_WINDOW, 800, 600);
//
//    // 设置鼠标回调
//    cv::setMouseCallback(MAIN_WINDOW, onMouseCallback, this);
//    cv::setMouseCallback(RESULT_WINDOW, onMouseCallback, this);
//}
//
//void GUIInterface::createTrackbars() {
//    // 在控制面板创建滑块
//    cv::createTrackbar("窗口大小", CONTROL_WINDOW, &m_windowSize, 31, onWindowSizeChange, this);
//    cv::setTrackbarMin("窗口大小", CONTROL_WINDOW, 5);
//
//    cv::createTrackbar("水体同质性阈值", CONTROL_WINDOW, &m_waterHomoThresh, 255, onWaterHomoChange, this);
//    cv::createTrackbar("水体对比度阈值", CONTROL_WINDOW, &m_waterContrastThresh, 255, onWaterContrastChange, this);
//    cv::createTrackbar("林地同质性阈值", CONTROL_WINDOW, &m_forestHomoThresh, 255, onForestHomoChange, this);
//    cv::createTrackbar("林地对比度阈值", CONTROL_WINDOW, &m_forestContrastThresh, 255, onForestContrastChange, this);
//    cv::createTrackbar("DNN增强", CONTROL_WINDOW, &m_useDNN, 1, onDNNToggle, this);
//    cv::createTrackbar("显示模式", CONTROL_WINDOW, &m_displayMode, 5, onDisplayModeChange, this);
//
//    // 创建控制面板背景
//    cv::Mat controlBg = cv::Mat::zeros(450, 400, CV_8UC3);
//    controlBg.setTo(cv::Scalar(50, 50, 50));
//
//    // 添加说明文字
//    int y = 30;
//    cv::putText(controlBg, "=== GLCM Water/Forest Extraction ===",
//        cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
//
//    y += 40;
//    cv::putText(controlBg, "Parameters:", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    y += 25;
//    cv::putText(controlBg, "- Window Size: GLCM calculation window",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 20;
//    cv::putText(controlBg, "- Water Homo: High=more water detected",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 20;
//    cv::putText(controlBg, "- Water Contrast: Low=smoother water",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 20;
//    cv::putText(controlBg, "- Forest Homo: Low=more texture",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 20;
//    cv::putText(controlBg, "- Forest Contrast: High=more edges",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 40;
//    cv::putText(controlBg, "Display Modes:", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    y += 25;
//    cv::putText(controlBg, "0: Original | 1: Grayscale | 2: Contrast",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 20;
//    cv::putText(controlBg, "3: Homogeneity | 4: Water | 5: Forest",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
//
//    y += 40;
//    cv::putText(controlBg, "Keyboard Shortcuts:", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    y += 25;
//    cv::putText(controlBg, "[P] Process | [S] Save | [H] Help | [Q] Quit",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
//
//    y += 20;
//    cv::putText(controlBg, "[1-3] Load cloudy/mss/pan.bmp",
//        cv::Point(30, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
//
//    cv::imshow(CONTROL_WINDOW, controlBg);
//}
//
//bool GUIInterface::loadImage(const std::string& path) {
//    m_originalImage = cv::imread(path);
//    if (m_originalImage.empty()) {
//        std::cerr << "无法加载图像: " << path << std::endl;
//        return false;
//    }
//
//    m_currentImagePath = path;
//    m_displayImage = m_originalImage.clone();
//    m_results.clear();
//
//    std::cout << "已加载图像: " << path << std::endl;
//    std::cout << "图像尺寸: " << m_originalImage.cols << " x " << m_originalImage.rows << std::endl;
//
//    updateDisplay();
//    return true;
//}
//
//void GUIInterface::processCurrentImage() {
//    if (m_originalImage.empty()) {
//        std::cout << "请先加载图像!" << std::endl;
//        return;
//    }
//
//    std::cout << "\n开始处理图像..." << std::endl;
//    auto startTime = std::chrono::high_resolution_clock::now();
//
//    // 确保窗口大小为奇数
//    int windowSize = m_windowSize;
//    if (windowSize % 2 == 0) windowSize++;
//
//    // 设置参数
//    std::map<std::string, int> params;
//    params["water_homo"] = m_waterHomoThresh;
//    params["water_contrast"] = m_waterContrastThresh;
//    params["forest_homo"] = m_forestHomoThresh;
//    params["forest_contrast"] = m_forestContrastThresh;
//
//    try {
//        if (m_useDNN) {
//            m_results = m_extractor.processImageEnhanced(
//                m_currentImagePath, windowSize, params, true);
//        }
//        else {
//            m_results = m_extractor.processImage(
//                m_currentImagePath, windowSize, params);
//        }
//
//        auto endTime = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//        std::cout << "处理完成，耗时: " << duration.count() << " ms" << std::endl;
//
//        updateDisplay();
//        updateFeatureDisplay();
//
//    }
//    catch (const std::exception& e) {
//        std::cerr << "处理出错: " << e.what() << std::endl;
//    }
//}
//
//void GUIInterface::saveResults(const std::string& basePath) {
//    if (m_results.empty()) {
//        std::cout << "没有结果可保存，请先处理图像!" << std::endl;
//        return;
//    }
//
//    std::string base = basePath;
//    if (base.empty()) {
//        base = m_currentImagePath.substr(0, m_currentImagePath.find_last_of("."));
//    }
//
//    // 保存各种结果
//    if (m_results.count("overlay")) {
//        cv::imwrite(base + "_overlay.png", m_results["overlay"]);
//    }
//    if (m_results.count("water_mask")) {
//        cv::Mat waterVis;
//        m_results["water_mask"].convertTo(waterVis, CV_8U, 255);
//        cv::imwrite(base + "_water.png", waterVis);
//    }
//    if (m_results.count("forest_mask")) {
//        cv::Mat forestVis;
//        m_results["forest_mask"].convertTo(forestVis, CV_8U, 255);
//        cv::imwrite(base + "_forest.png", forestVis);
//    }
//    if (m_results.count("contrast")) {
//        cv::Mat contrastVis;
//        cv::normalize(m_results["contrast"], contrastVis, 0, 255, cv::NORM_MINMAX, CV_8U);
//        cv::applyColorMap(contrastVis, contrastVis, cv::COLORMAP_JET);
//        cv::imwrite(base + "_contrast.png", contrastVis);
//    }
//    if (m_results.count("homogeneity")) {
//        cv::Mat homoVis;
//        cv::normalize(m_results["homogeneity"], homoVis, 0, 255, cv::NORM_MINMAX, CV_8U);
//        cv::applyColorMap(homoVis, homoVis, cv::COLORMAP_JET);
//        cv::imwrite(base + "_homogeneity.png", homoVis);
//    }
//
//    std::cout << "结果已保存到: " << base << "_*.png" << std::endl;
//}
//
//void GUIInterface::updateDisplay() {
//    if (m_originalImage.empty()) return;
//
//    cv::Mat display;
//
//    if (m_results.empty()) {
//        display = m_originalImage.clone();
//    }
//    else {
//        switch (m_displayMode) {
//        case 0: // 原始图像
//            display = m_results["original"].clone();
//            break;
//        case 1: // 灰度图
//            cv::cvtColor(m_results["gray"], display, cv::COLOR_GRAY2BGR);
//            break;
//        case 2: // 对比度
//            if (m_results.count("contrast")) {
//                cv::Mat vis;
//                cv::normalize(m_results["contrast"], vis, 0, 255, cv::NORM_MINMAX, CV_8U);
//                cv::applyColorMap(vis, display, cv::COLORMAP_JET);
//            }
//            break;
//        case 3: // 同质性
//            if (m_results.count("homogeneity")) {
//                cv::Mat vis;
//                cv::normalize(m_results["homogeneity"], vis, 0, 255, cv::NORM_MINMAX, CV_8U);
//                cv::applyColorMap(vis, display, cv::COLORMAP_JET);
//            }
//            break;
//        case 4: // 水体
//            if (m_results.count("water_mask")) {
//                cv::Mat vis;
//                m_results["water_mask"].convertTo(vis, CV_8U, 255);
//                cv::applyColorMap(vis, display, cv::COLORMAP_WINTER);
//            }
//            break;
//        case 5: // 林地
//            if (m_results.count("forest_mask")) {
//                cv::Mat vis;
//                m_results["forest_mask"].convertTo(vis, CV_8U, 255);
//                cv::applyColorMap(vis, display, cv::COLORMAP_SUMMER);
//            }
//            break;
//        default:
//            display = m_results.count("overlay") ? m_results["overlay"].clone() : m_originalImage.clone();
//        }
//    }
//
//    if (display.empty()) {
//        display = m_originalImage.clone();
//    }
//
//    // 绘制状态栏
//    drawStatusBar(display);
//
//    cv::imshow(MAIN_WINDOW, display);
//
//    // 更新结果窗口
//    if (!m_results.empty() && m_results.count("overlay")) {
//        cv::imshow(RESULT_WINDOW, createResultVisualization());
//    }
//}
//
//void GUIInterface::updateFeatureDisplay() {
//    if (m_results.empty()) return;
//
//    cv::Mat featureVis = createFeatureVisualization();
//    if (!featureVis.empty()) {
//        cv::imshow(FEATURE_WINDOW, featureVis);
//    }
//}
//
//void GUIInterface::drawStatusBar(cv::Mat& img) {
//    int barHeight = 30;
//    cv::Rect barRect(0, img.rows - barHeight, img.cols, barHeight);
//
//    // 半透明背景
//    cv::Mat roi = img(barRect);
//    cv::Mat overlay = cv::Mat::zeros(roi.size(), roi.type());
//    overlay.setTo(cv::Scalar(0, 0, 0));
//    cv::addWeighted(roi, 0.5, overlay, 0.5, 0, roi);
//
//    // 状态文字
//    std::stringstream ss;
//    ss << "Image: " << m_currentImagePath
//        << " | Window: " << m_windowSize
//        << " | DNN: " << (m_useDNN ? "ON" : "OFF")
//        << " | Mode: " << m_displayMode;
//
//    cv::putText(img, ss.str(), cv::Point(10, img.rows - 10),
//        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
//}
//
//cv::Mat GUIInterface::createFeatureVisualization() {
//    if (m_results.empty()) return cv::Mat();
//
//    int targetH = 300;
//    int targetW = 400;
//
//    std::vector<cv::Mat> features;
//    std::vector<std::string> titles = { "Contrast", "Homogeneity", "Energy", "Entropy" };
//    std::vector<std::string> keys = { "contrast", "homogeneity", "energy", "entropy" };
//
//    for (size_t i = 0; i < keys.size(); i++) {
//        if (m_results.count(keys[i])) {
//            cv::Mat vis;
//            cv::normalize(m_results[keys[i]], vis, 0, 255, cv::NORM_MINMAX, CV_8U);
//            cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
//            cv::resize(vis, vis, cv::Size(targetW, targetH));
//
//            // 添加标题
//            cv::putText(vis, titles[i], cv::Point(10, 25),
//                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
//
//            features.push_back(vis);
//        }
//    }
//
//    if (features.empty()) return cv::Mat();
//
//    // 创建2x2网格
//    cv::Mat result = cv::Mat::zeros(targetH * 2, targetW * 2, CV_8UC3);
//
//    for (size_t i = 0; i < features.size() && i < 4; i++) {
//        int row = i / 2;
//        int col = i % 2;
//        cv::Rect roi(col * targetW, row * targetH, targetW, targetH);
//        features[i].copyTo(result(roi));
//    }
//
//    return result;
//}
//
//cv::Mat GUIInterface::createResultVisualization() {
//    if (m_results.empty()) return cv::Mat();
//
//    cv::Mat result;
//
//    if (m_results.count("overlay")) {
//        result = m_results["overlay"].clone();
//    }
//    else if (m_results.count("original")) {
//        result = m_results["original"].clone();
//    }
//    else {
//        return cv::Mat();
//    }
//
//    // 添加图例
//    cv::Mat legend = createLegend();
//
//    // 在结果图上叠加图例
//    if (!legend.empty() && result.cols > legend.cols && result.rows > legend.rows) {
//        cv::Rect legendRoi(result.cols - legend.cols - 10, 10, legend.cols, legend.rows);
//        legend.copyTo(result(legendRoi));
//    }
//
//    // 添加统计信息
//    if (m_results.count("water_mask") && m_results.count("forest_mask")) {
//        auto stats = m_extractor.computeStats(m_results["water_mask"], m_results["forest_mask"], 2.0);
//
//        std::stringstream ss;
//        ss << std::fixed << std::setprecision(2);
//        ss << "Water: " << stats.waterRatio * 100 << "% (" << stats.waterArea / 10000 << " ha)";
//        cv::putText(result, ss.str(), cv::Point(10, result.rows - 40),
//            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
//
//        ss.str("");
//        ss << "Forest: " << stats.forestRatio * 100 << "% (" << stats.forestArea / 10000 << " ha)";
//        cv::putText(result, ss.str(), cv::Point(10, result.rows - 15),
//            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//    }
//
//    return result;
//}
//
//cv::Mat GUIInterface::createLegend() {
//    int width = 150;
//    int height = 80;
//
//    cv::Mat legend = cv::Mat::zeros(height, width, CV_8UC3);
//    legend.setTo(cv::Scalar(50, 50, 50));
//
//    // 水体图例
//    cv::rectangle(legend, cv::Point(10, 15), cv::Point(30, 35), cv::Scalar(255, 0, 0), -1);
//    cv::putText(legend, "Water", cv::Point(40, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    // 林地图例
//    cv::rectangle(legend, cv::Point(10, 45), cv::Point(30, 65), cv::Scalar(0, 255, 0), -1);
//    cv::putText(legend, "Forest", cv::Point(40, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    return legend;
//}
//
//void GUIInterface::showHelp() {
//    cv::Mat helpImg = cv::Mat::zeros(400, 500, CV_8UC3);
//    helpImg.setTo(cv::Scalar(40, 40, 40));
//
//    int y = 30;
//    cv::putText(helpImg, "=== GLCM Water/Forest Extraction Help ===",
//        cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
//
//    y += 40;
//    cv::putText(helpImg, "Keyboard Controls:", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    std::vector<std::pair<std::string, std::string>> shortcuts = {
//        {"P", "Process current image"},
//        {"S", "Save results"},
//        {"H", "Show this help"},
//        {"Q/ESC", "Quit application"},
//        {"1", "Load cloudy.bmp"},
//        {"2", "Load mss.bmp"},
//        {"3", "Load pan.bmp"},
//        {"0-5", "Change display mode"}
//    };
//
//    y += 25;
//    for (const auto& sc : shortcuts) {
//        cv::putText(helpImg, "[" + sc.first + "] " + sc.second, cv::Point(40, y),
//            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
//        y += 22;
//    }
//
//    y += 20;
//    cv::putText(helpImg, "Parameter Tips:", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//
//    y += 25;
//    cv::putText(helpImg, "Water: High homogeneity + Low contrast", cv::Point(40, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 200, 255), 1);
//
//    y += 22;
//    cv::putText(helpImg, "Forest: Low homogeneity + High contrast", cv::Point(40, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 255, 100), 1);
//
//    y += 35;
//    cv::putText(helpImg, "Press any key to close...", cv::Point(20, y),
//        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
//
//    cv::imshow("Help", helpImg);
//    cv::waitKey(0);
//    cv::destroyWindow("Help");
//}
//
//void GUIInterface::run(const std::string& imagePath) {
//    createWindows();
//    createTrackbars();
//
//    if (!imagePath.empty()) {
//        loadImage(imagePath);
//    }
//
//    std::cout << "\n=== GLCM水体/林地提取系统 ===" << std::endl;
//    std::cout << "按键操作:" << std::endl;
//    std::cout << "  P - 处理图像" << std::endl;
//    std::cout << "  S - 保存结果" << std::endl;
//    std::cout << "  H - 显示帮助" << std::endl;
//    std::cout << "  1/2/3 - 加载 cloudy/mss/pan.bmp" << std::endl;
//    std::cout << "  Q/ESC - 退出" << std::endl;
//
//    while (true) {
//        int key = cv::waitKey(50);
//
//        if (key == 'q' || key == 'Q' || key == 27) {
//            break;
//        }
//        else if (key == 'p' || key == 'P') {
//            processCurrentImage();
//        }
//        else if (key == 's' || key == 'S') {
//            saveResults("");
//        }
//        else if (key == 'h' || key == 'H') {
//            showHelp();
//        }
//        else if (key == '1') {
//            loadImage("cloudy.bmp");
//        }
//        else if (key == '2') {
//            loadImage("mss.bmp");
//        }
//        else if (key == '3') {
//            loadImage("pan.bmp");
//        }
//        else if (key >= '0' && key <= '5') {
//            m_displayMode = key - '0';
//            cv::setTrackbarPos("显示模式", CONTROL_WINDOW, m_displayMode);
//            updateDisplay();
//        }
//    }
//
//    cv::destroyAllWindows();
//}
//
//// 回调函数实现
//void GUIInterface::onWindowSizeChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_windowSize = value;
//    if (gui->m_windowSize < 5) gui->m_windowSize = 5;
//    if (gui->m_windowSize % 2 == 0) gui->m_windowSize++;
//}
//
//void GUIInterface::onWaterHomoChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_waterHomoThresh = value;
//}
//
//void GUIInterface::onWaterContrastChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_waterContrastThresh = value;
//}
//
//void GUIInterface::onForestHomoChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_forestHomoThresh = value;
//}
//
//void GUIInterface::onForestContrastChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_forestContrastThresh = value;
//}
//
//void GUIInterface::onDNNToggle(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_useDNN = value;
//}
//
//void GUIInterface::onDisplayModeChange(int value, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//    gui->m_displayMode = value;
//    gui->updateDisplay();
//}
//
//void GUIInterface::onMouseCallback(int event, int x, int y, int flags, void* userdata) {
//    GUIInterface* gui = static_cast<GUIInterface*>(userdata);
//
//    if (event == cv::EVENT_LBUTTONDOWN) {
//        // 显示点击位置的像素信息
//        if (!gui->m_originalImage.empty() &&
//            x >= 0 && x < gui->m_originalImage.cols &&
//            y >= 0 && y < gui->m_originalImage.rows) {
//
//            cv::Vec3b pixel = gui->m_originalImage.at<cv::Vec3b>(y, x);
//            std::cout << "位置 (" << x << ", " << y << "): "
//                << "B=" << (int)pixel[0]
//                << " G=" << (int)pixel[1]
//                << " R=" << (int)pixel[2] << std::endl;
//
//            // 如果有特征图，也显示特征值
//            if (gui->m_results.count("contrast") &&
//                y < gui->m_results["contrast"].rows &&
//                x < gui->m_results["contrast"].cols) {
//                double contrast = gui->m_results["contrast"].at<double>(y, x);
//                double homo = gui->m_results["homogeneity"].at<double>(y, x);
//                std::cout << "  Contrast: " << contrast << ", Homogeneity: " << homo << std::endl;
//            }
//        }
//    }
//}
















/**
 * GUI界面实现
 * 文件名: GUIInterface.cpp
 */

#include "GUIInterface.h"
#include <iostream>
#include <sstream>
#include <iomanip>

const std::string GUIInterface::MAIN_WINDOW = "GLCM Extractor";
const std::string GUIInterface::CONTROL_WINDOW = "Controls";

GUIInterface::GUIInterface()
    : m_blockSize(8),
    m_minArea(100),
    m_useAdaptive(1),
    m_waterHomoThresh(180),
    m_waterContrastThresh(50),
    m_forestHomoThresh(120),
    m_forestContrastThresh(80),
    m_displayMode(0),
    m_hasResult(false) {
}

GUIInterface::~GUIInterface() {
    cv::destroyAllWindows();
}

void GUIInterface::createWindows() {
    cv::namedWindow(MAIN_WINDOW, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(CONTROL_WINDOW, cv::WINDOW_AUTOSIZE);
}

void GUIInterface::createTrackbars() {
    // 控制面板背景
    cv::Mat panel(400, 400, CV_8UC3, cv::Scalar(45, 45, 50));

    cv::putText(panel, "GLCM Water/Forest Extractor", cv::Point(50, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 200, 80), 1);

    cv::line(panel, cv::Point(20, 50), cv::Point(380, 50), cv::Scalar(80, 80, 85), 1);

    cv::putText(panel, "[1][2][3] Load cloudy/mss/pan.bmp", cv::Point(20, 75),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 255, 150), 1);

    cv::putText(panel, "[P] Process  [S] Save  [Q] Quit", cv::Point(20, 100),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 255, 150), 1);

    cv::line(panel, cv::Point(20, 120), cv::Point(380, 120), cv::Scalar(80, 80, 85), 1);

    cv::putText(panel, "Adaptive=1: Auto threshold (recommended)", cv::Point(20, 145),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 200, 255), 1);

    cv::putText(panel, "Adaptive=0: Use manual thresholds below", cv::Point(20, 165),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);

    cv::line(panel, cv::Point(20, 185), cv::Point(380, 185), cv::Scalar(80, 80, 85), 1);

    cv::putText(panel, "Water: High Homo + Low Contrast", cv::Point(20, 210),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 180, 100), 1);

    cv::putText(panel, "Forest: Low Homo + High Contrast", cv::Point(20, 230),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 255, 150), 1);

    cv::imshow(CONTROL_WINDOW, panel);

    // 创建滑动条
    cv::createTrackbar("BlockSize", CONTROL_WINDOW, &m_blockSize, 32);
    cv::createTrackbar("MinArea", CONTROL_WINDOW, &m_minArea, 500);
    cv::createTrackbar("Adaptive", CONTROL_WINDOW, &m_useAdaptive, 1);
    cv::createTrackbar("WaterHomo", CONTROL_WINDOW, &m_waterHomoThresh, 255);
    cv::createTrackbar("WaterContrast", CONTROL_WINDOW, &m_waterContrastThresh, 255);
    cv::createTrackbar("ForestHomo", CONTROL_WINDOW, &m_forestHomoThresh, 255);
    cv::createTrackbar("ForestContrast", CONTROL_WINDOW, &m_forestContrastThresh, 255);
}

bool GUIInterface::loadImage(const std::string& path) {
    m_originalImage = cv::imread(path);
    if (m_originalImage.empty()) {
        std::cout << "Cannot load: " << path << std::endl;
        return false;
    }

    m_currentImagePath = path;
    m_hasResult = false;

    std::cout << "\nLoaded: " << path << " (" << m_originalImage.cols << "x" << m_originalImage.rows << ")" << std::endl;

    updateDisplay();
    return true;
}

void GUIInterface::processCurrentImage() {
    if (m_originalImage.empty()) {
        std::cout << "Load an image first!" << std::endl;
        return;
    }

    GLCMExtractor::Params params;
    params.blockSize = std::max(4, m_blockSize);
    params.minArea = m_minArea;
    params.useAdaptiveThresh = (m_useAdaptive == 1);
    params.waterHomoMin = m_waterHomoThresh;
    params.waterContrastMax = m_waterContrastThresh;
    params.forestHomoMax = m_forestHomoThresh;
    params.forestContrastMin = m_forestContrastThresh;

    std::cout << "\nProcessing with BlockSize=" << params.blockSize
        << ", Adaptive=" << (params.useAdaptiveThresh ? "ON" : "OFF") << std::endl;

    m_result = m_extractor.process(m_originalImage, params);
    m_hasResult = true;

    updateDisplay();
}

void GUIInterface::saveResults(const std::string& basePath) {
    if (!m_hasResult) {
        std::cout << "Process an image first!" << std::endl;
        return;
    }

    std::string base = basePath;
    if (base.empty()) {
        base = m_currentImagePath.substr(0, m_currentImagePath.find_last_of("."));
    }

    cv::imwrite(base + "_overlay.png", m_result.overlay);
    cv::imwrite(base + "_water.png", m_result.waterMask);
    cv::imwrite(base + "_forest.png", m_result.forestMask);

    // 保存特征图
    cv::Mat contrastVis, homoVis;
    cv::normalize(m_result.contrast, contrastVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(m_result.homogeneity, homoVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(contrastVis, contrastVis, cv::COLORMAP_JET);
    cv::applyColorMap(homoVis, homoVis, cv::COLORMAP_JET);
    cv::imwrite(base + "_contrast.png", contrastVis);
    cv::imwrite(base + "_homogeneity.png", homoVis);

    std::cout << "\nSaved:" << std::endl;
    std::cout << "  " << base << "_overlay.png" << std::endl;
    std::cout << "  " << base << "_water.png" << std::endl;
    std::cout << "  " << base << "_forest.png" << std::endl;
    std::cout << "  " << base << "_contrast.png" << std::endl;
    std::cout << "  " << base << "_homogeneity.png" << std::endl;
}

void GUIInterface::updateDisplay() {
    if (m_originalImage.empty()) return;

    cv::Mat display;

    if (m_hasResult) {
        // 根据显示模式选择
        switch (m_displayMode) {
        case 1: {
            // 对比度特征图
            cv::Mat vis;
            cv::normalize(m_result.contrast, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(vis, display, cv::COLORMAP_JET);
            break;
        }
        case 2: {
            // 同质性特征图
            cv::Mat vis;
            cv::normalize(m_result.homogeneity, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(vis, display, cv::COLORMAP_JET);
            break;
        }
        case 3: {
            // 水体掩膜
            cv::cvtColor(m_result.waterMask, display, cv::COLOR_GRAY2BGR);
            break;
        }
        case 4: {
            // 林地掩膜
            cv::cvtColor(m_result.forestMask, display, cv::COLOR_GRAY2BGR);
            break;
        }
        default:
            display = m_result.overlay.clone();
            break;
        }

        // 添加图例（仅在overlay模式）
        if (m_displayMode == 0) {
            cv::rectangle(display, cv::Point(10, 10), cv::Point(130, 75), cv::Scalar(30, 30, 35), -1);
            cv::rectangle(display, cv::Point(10, 10), cv::Point(130, 75), cv::Scalar(100, 100, 100), 1);

            cv::rectangle(display, cv::Point(15, 18), cv::Point(30, 28), cv::Scalar(255, 150, 0), -1);
            cv::putText(display, "Water", cv::Point(35, 27), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

            cv::rectangle(display, cv::Point(15, 35), cv::Point(30, 45), cv::Scalar(0, 200, 50), -1);
            cv::putText(display, "Forest", cv::Point(35, 44), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << "W:" << m_result.waterPercent << "% F:" << m_result.forestPercent << "%";
            cv::putText(display, ss.str(), cv::Point(15, 68), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
        }

        // 显示模式标签
        std::string modeLabel;
        switch (m_displayMode) {
        case 1: modeLabel = "[Contrast]"; break;
        case 2: modeLabel = "[Homogeneity]"; break;
        case 3: modeLabel = "[Water Mask]"; break;
        case 4: modeLabel = "[Forest Mask]"; break;
        default: modeLabel = "[Overlay]"; break;
        }
        cv::putText(display, modeLabel, cv::Point(display.cols - 120, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
    else {
        display = m_originalImage.clone();
    }

    // 缩放显示
    double scale = std::min(1000.0 / display.cols, 750.0 / display.rows);
    if (scale < 1.0) {
        cv::resize(display, display, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::imshow(MAIN_WINDOW, display);
}

void GUIInterface::showHelp() {
    std::cout << "\n========== Help ==========" << std::endl;
    std::cout << "[P] Process image" << std::endl;
    std::cout << "[S] Save results" << std::endl;
    std::cout << "[Q/ESC] Quit" << std::endl;
    std::cout << "[1][2][3] Load images" << std::endl;
    std::cout << "[0-4] Display mode:" << std::endl;
    std::cout << "  0=Overlay, 1=Contrast, 2=Homogeneity" << std::endl;
    std::cout << "  3=Water, 4=Forest" << std::endl;
    std::cout << "==========================\n" << std::endl;
}

void GUIInterface::run(const std::string& imagePath) {
    createWindows();
    createTrackbars();

    if (!imagePath.empty()) {
        loadImage(imagePath);
    }
    else {
        cv::Mat welcome(500, 700, CV_8UC3, cv::Scalar(40, 40, 45));

        cv::putText(welcome, "GLCM Water/Forest Extractor", cv::Point(150, 180),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 200, 80), 2);

        cv::putText(welcome, "Based on Gray-Level Co-occurrence Matrix", cv::Point(165, 220),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1);

        cv::line(welcome, cv::Point(150, 250), cv::Point(550, 250), cv::Scalar(80, 80, 85), 1);

        cv::putText(welcome, "Press [1][2][3] to load image", cv::Point(230, 290),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 255, 150), 1);

        cv::putText(welcome, "[P] Process  [S] Save  [Q] Quit", cv::Point(220, 330),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 255, 150), 1);

        cv::putText(welcome, "[0-4] Change display mode", cv::Point(245, 370),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imshow(MAIN_WINDOW, welcome);
    }

    std::cout << "\n===== GLCM Extractor =====" << std::endl;
    std::cout << "Press [H] for help" << std::endl;

    while (true) {
        int key = cv::waitKey(30);
        if (key == -1) continue;

        if (key == 'q' || key == 'Q' || key == 27) break;
        else if (key == 'p' || key == 'P') processCurrentImage();
        else if (key == 's' || key == 'S') saveResults("");
        else if (key == 'h' || key == 'H') showHelp();
        else if (key == '1') loadImage("cloudy.bmp");
        else if (key == '2') loadImage("mss.bmp");
        else if (key == '3') loadImage("pan.bmp");
        else if (key >= '0' && key <= '4') {
            m_displayMode = key - '0';
            if (m_hasResult) updateDisplay();
        }
    }

    cv::destroyAllWindows();
}