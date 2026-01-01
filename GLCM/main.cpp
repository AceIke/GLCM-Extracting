///**
// * @file main.cpp
// * @brief GLCM水体/林地提取系统主程序
// *
// * 数字图像处理课程设计 - 任务1
// * 基于灰度共生矩阵的水体/林地提取
// *
// * 功能特点：
// * 1. GLCM纹理特征提取（对比度、同质性、能量、相关性、熵）
// * 2. DNN增强特征提取（可选）
// * 3. Gabor滤波器纹理增强
// * 4. 自适应阈值分割
// * 5. 形态学后处理
// * 6. 图形用户界面（基于OpenCV HighGUI）
// *
// * 使用方法：
// *   - 命令行模式: ./glcm_extractor <image_path>
// *   - GUI模式: ./glcm_extractor --gui [image_path]
// *   - 批处理模式: ./glcm_extractor --batch
// */
//
//#include "GLCMExtractor.h"
//#include "GUIInterface.h"
//#include <iostream>
//#include <vector>
//#include <string>
//#include <chrono>
//
// // 打印使用说明
//void printUsage(const char* progName) {
//    std::cout << "\n使用方法:\n";
//    std::cout << "  " << progName << " <image_path>           处理单张图像\n";
//    std::cout << "  " << progName << " --gui [image_path]     启动GUI模式\n";
//    std::cout << "  " << progName << " --batch                 批量处理所有图像\n";
//    std::cout << "  " << progName << " --help                  显示帮助信息\n";
//    std::cout << "\n示例:\n";
//    std::cout << "  " << progName << " cloudy.bmp\n";
//    std::cout << "  " << progName << " --gui pan.bmp\n";
//    std::cout << "  " << progName << " --batch\n";
//}
//
//// 批量处理模式
//void batchProcess() {
//    std::cout << "\n========== 批量处理模式 ==========\n";
//
//    // 初始化提取器
//    std::vector<int> distances = { 1, 3, 5 };
//    std::vector<double> angles = { 0, 45, 90, 135 };
//    GLCMExtractor extractor(distances, angles, 256);
//
//    // 要处理的图像列表
//    std::vector<std::string> imageFiles = {
//        "cloudy.bmp",
//        "mss.bmp",
//        "pan.bmp"
//    };
//
//    // 分割参数
//    std::map<std::string, int> params;
//    params["water_homo"] = 200;
//    params["water_contrast"] = 50;
//    params["forest_homo"] = 150;
//    params["forest_contrast"] = 80;
//
//    int windowSize = 15;
//
//    // 处理每张图像
//    for (const auto& imgFile : imageFiles) {
//        std::cout << "\n----------------------------------------\n";
//        std::cout << "处理图像: " << imgFile << "\n";
//        std::cout << "----------------------------------------\n";
//
//        try {
//            auto startTime = std::chrono::high_resolution_clock::now();
//
//            // 使用DNN增强处理
//            auto results = extractor.processImageEnhanced(imgFile, windowSize, params, true);
//
//            auto endTime = std::chrono::high_resolution_clock::now();
//            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//
//            // 保存结果
//            std::string savePath = imgFile.substr(0, imgFile.find_last_of(".")) + "_result.png";
//            extractor.visualizeResults(results, savePath);
//
//            std::cout << imgFile << " 处理完成！耗时: " << duration.count() << " ms\n";
//        }
//        catch (const std::exception& e) {
//            std::cerr << "处理 " << imgFile << " 时出错: " << e.what() << "\n";
//            continue;
//        }
//    }
//
//    std::cout << "\n========== 所有图像处理完成！ ==========\n";
//}
//
//// 单图像处理模式
//void singleProcess(const std::string& imagePath) {
//    std::cout << "\n========== 单图像处理模式 ==========\n";
//
//    std::vector<int> distances = { 1, 3, 5 };
//    std::vector<double> angles = { 0, 45, 90, 135 };
//    GLCMExtractor extractor(distances, angles, 256);
//
//    std::map<std::string, int> params;
//    params["water_homo"] = 200;
//    params["water_contrast"] = 50;
//    params["forest_homo"] = 150;
//    params["forest_contrast"] = 80;
//
//    try {
//        auto startTime = std::chrono::high_resolution_clock::now();
//
//        auto results = extractor.processImageEnhanced(imagePath, 15, params, true);
//
//        auto endTime = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//
//        std::string savePath = imagePath.substr(0, imagePath.find_last_of(".")) + "_result.png";
//        extractor.visualizeResults(results, savePath);
//
//        std::cout << "\n处理完成！耗时: " << duration.count() << " ms\n";
//    }
//    catch (const std::exception& e) {
//        std::cerr << "处理出错: " << e.what() << "\n";
//    }
//}
//
//// GUI模式
//void guiMode(const std::string& imagePath = "") {
//    std::cout << "\n========== GUI模式 ==========\n";
//
//    GUIInterface gui;
//    gui.run(imagePath);
//}
//
//int main(int argc, char** argv) {
//    std::cout << "=========================================\n";
//    std::cout << "  GLCM水体/林地提取系统 (DNN增强版)\n";
//    std::cout << "  数字图像处理课程设计 - 任务1\n";
//    std::cout << "=========================================\n";
//
//    try {
//        if (argc < 2) {
//            // 默认进入GUI模式
//            guiMode();
//        }
//        else {
//            std::string arg1 = argv[1];
//
//            if (arg1 == "--help" || arg1 == "-h") {
//                printUsage(argv[0]);
//            }
//            else if (arg1 == "--gui") {
//                if (argc > 2) {
//                    guiMode(argv[2]);
//                }
//                else {
//                    guiMode();
//                }
//            }
//            else if (arg1 == "--batch") {
//                batchProcess();
//            }
//            else {
//                // 单图像处理
//                singleProcess(arg1);
//            }
//        }
//    }
//    catch (const std::exception& e) {
//        std::cerr << "程序运行错误: " << e.what() << "\n";
//        return -1;
//    }
//
//    return 0;
//}


















/**
 * GLCM水体/林地提取系统 - 主程序
 * 文件名: main.cpp
 *
 * 使用方法:
 *   程序名                    启动GUI
 *   程序名 image.bmp          处理单张图像
 *   程序名 --gui image.bmp    GUI模式加载图像
 */

#include "GLCMextractor.h"
#include "GUIInterface.h"
#include <iostream>

void processAndSave(const std::string& imagePath) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing: " << imagePath << std::endl;
    std::cout << "========================================" << std::endl;

    GLCMExtractor extractor;
    GLCMExtractor::Params params;
    params.blockSize = 16;
    params.grayLevels = 8;

    auto result = extractor.process(imagePath, params);

    std::string base = imagePath.substr(0, imagePath.find_last_of("."));
    cv::imwrite(base + "_overlay.png", result.overlay);
    cv::imwrite(base + "_water.png", result.waterMask);
    cv::imwrite(base + "_forest.png", result.forestMask);

    std::cout << "\nSaved:" << std::endl;
    std::cout << "  " << base << "_overlay.png" << std::endl;
    std::cout << "  " << base << "_water.png" << std::endl;
    std::cout << "  " << base << "_forest.png" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GLCM Water/Forest Extractor" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        if (argc == 1) {
            // 无参数，启动GUI
            GUIInterface gui;
            gui.run();
        }
        else if (argc == 2) {
            std::string arg = argv[1];
            if (arg == "--help" || arg == "-h") {
                std::cout << "\nUsage:" << std::endl;
                std::cout << "  " << argv[0] << "                Start GUI" << std::endl;
                std::cout << "  " << argv[0] << " image.bmp      Process image" << std::endl;
                std::cout << "  " << argv[0] << " --gui [image]  GUI with image" << std::endl;
            }
            else if (arg == "--gui") {
                GUIInterface gui;
                gui.run();
            }
            else {
                processAndSave(arg);
            }
        }
        else if (argc >= 3 && std::string(argv[1]) == "--gui") {
            GUIInterface gui;
            gui.run(argv[2]);
        }
        else {
            processAndSave(argv[1]);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}