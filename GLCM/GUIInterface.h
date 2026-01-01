//#ifndef GUI_INTERFACE_H
//#define GUI_INTERFACE_H
//
//#include "GLCMExtractor.h"
//#include <opencv2/opencv.hpp>
//#include <string>
//#include <map>
//#include <functional>
//
///**
// * @brief 基于OpenCV HighGUI的图形用户界面
// *
// * 不依赖Qt，使用OpenCV内置的GUI功能：
// * - 窗口管理
// * - Trackbar（滑块）参数调节
// * - 鼠标回调
// * - 按键事件处理
// */
//class GUIInterface {
//public:
//    GUIInterface();
//    ~GUIInterface();
//
//    /**
//     * @brief 运行GUI主循环
//     * @param imagePath 默认图像路径（可选）
//     */
//    void run(const std::string& imagePath = "");
//
//    /**
//     * @brief 加载图像
//     * @param path 图像路径
//     * @return 是否成功
//     */
//    bool loadImage(const std::string& path);
//
//    /**
//     * @brief 处理当前图像
//     */
//    void processCurrentImage();
//
//    /**
//     * @brief 保存结果
//     * @param basePath 基础路径
//     */
//    void saveResults(const std::string& basePath);
//
//private:
//    // 参数
//    int m_windowSize;
//    int m_waterHomoThresh;
//    int m_waterContrastThresh;
//    int m_forestHomoThresh;
//    int m_forestContrastThresh;
//    int m_useDNN;
//
//    // 图像数据
//    cv::Mat m_originalImage;
//    cv::Mat m_displayImage;
//    std::map<std::string, cv::Mat> m_results;
//    std::string m_currentImagePath;
//
//    // GLCM提取器
//    GLCMExtractor m_extractor;
//
//    // 显示模式
//    int m_displayMode;
//
//    // 窗口名称
//    static const std::string MAIN_WINDOW;
//    static const std::string CONTROL_WINDOW;
//    static const std::string RESULT_WINDOW;
//    static const std::string FEATURE_WINDOW;
//
//    // 回调函数
//    static void onWindowSizeChange(int value, void* userdata);
//    static void onWaterHomoChange(int value, void* userdata);
//    static void onWaterContrastChange(int value, void* userdata);
//    static void onForestHomoChange(int value, void* userdata);
//    static void onForestContrastChange(int value, void* userdata);
//    static void onDNNToggle(int value, void* userdata);
//    static void onDisplayModeChange(int value, void* userdata);
//    static void onMouseCallback(int event, int x, int y, int flags, void* userdata);
//
//    // 辅助函数
//    void createWindows();
//    void createTrackbars();
//    void updateDisplay();
//    void updateFeatureDisplay();
//    void drawStatusBar(cv::Mat& img);
//    void showHelp();
//    cv::Mat createFeatureVisualization();
//    cv::Mat createResultVisualization();
//    cv::Mat createLegend();
//};
//
//#endif // GUI_INTERFACE_H#pragma once

















/**
 * GUI界面 - 只有声明
 * 文件名: GUIInterface.h
 */

#pragma once
#ifndef GUI_INTERFACE_H
#define GUI_INTERFACE_H

#include "GLCMextractor.h"
#include <string>

class GUIInterface {
public:
    GUIInterface();
    ~GUIInterface();

    void run(const std::string& imagePath = "");
    bool loadImage(const std::string& path);
    void processCurrentImage();
    void saveResults(const std::string& basePath);

private:
    // 参数
    int m_blockSize;
    int m_minArea;
    int m_useAdaptive;
    int m_waterHomoThresh;
    int m_waterContrastThresh;
    int m_forestHomoThresh;
    int m_forestContrastThresh;
    int m_displayMode;

    // 数据
    cv::Mat m_originalImage;
    GLCMExtractor::Result m_result;
    std::string m_currentImagePath;
    bool m_hasResult;

    // 提取器
    GLCMExtractor m_extractor;

    // 窗口名称
    static const std::string MAIN_WINDOW;
    static const std::string CONTROL_WINDOW;

    // 辅助函数
    void createWindows();
    void createTrackbars();
    void updateDisplay();
    void showHelp();
};

#endif // GUI_INTERFACE_H