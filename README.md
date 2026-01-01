# GLCM-Extracting
📖 项目简介
本项目是数字图像处理课程设计作品，实现了基于**灰度共生矩阵(GLCM)**的纹理特征提取与交互式可视化系统。用户可以通过GUI界面实时调整参数，观察不同纹理特征的提取效果。
✨ 主要功能

🖼️ GLCM纹理分析：计算灰度共生矩阵并提取四种关键纹理特征
🎛️ 实时参数调节：通过滑条实时调整dx、dy、灰度级等参数
🖱️ ROI区域选择：鼠标框选感兴趣区域进行局部分析
📊 特征可视化：实时显示能量、对比度、相关性、熵四种特征图
💾 结果导出：支持将特征值导出为CSV文件

🔬 算法原理
GLCM灰度共生矩阵
灰度共生矩阵(Gray Level Co-occurrence Matrix)统计图像中像素对的灰度联合分布，是经典的纹理分析方法(Haralick, 1973)。
矩阵定义：给定位移向量(dx, dy)，统计满足以下条件的像素对数量：

像素(x, y)灰度值为i
像素(x+dx, y+dy)灰度值为j

提取的四种纹理特征
特征公式物理意义能量(Energy/ASM)∑∑P(i,j)²衡量灰度分布均匀性，值越大纹理越均匀对比度(Contrast)∑∑(i-j)²P(i,j)衡量局部灰度变化强度，值越大纹理越粗糙相关性(Correlation)∑∑(i-μ)(j-μ)P(i,j)/σ²衡量行列方向线性相关性熵(Entropy)-∑∑P(i,j)log(P(i,j))衡量纹理随机性与复杂度

📁 项目结构
GLCM-Extracting/
├── GLCM/
│   ├── main.cpp              # 主程序入口
│   ├── GLCMextractor.h       # GLCM核心算法（Header-only）
│   ├── GLCMextracting.cpp     # GLCM算法实现
│   └── GUIInterface.h        # GUI交互接口
│   └── GUIInterface.cpp        # GUI交互接口
├── CMakeLists.txt            # CMake构建配置
└── README.md                 # 项目说明文档

🚀 快速开始
环境要求

操作系统：Windows 10/11
编译器：Visual Studio 2022 (支持C++17)
依赖库：OpenCV 4.x

方式一：Visual Studio(2022/2026)
克隆仓库
bashgit clone https://github.com/AceIke/GLCM-Extracting.git
cd GLCM-Extracting
配置OpenCV环境变量
OPENCV_DIR = C:\opencv\build
用Visual Studio打开项目，配置包含目录和库目录，编译运行

方式二：CMake
bashmkdir build && cd build
cmake ..
cmake --build . --config Release
./GLCM-Extracting

📄 License
本项目采用 MIT License 开源协议。

🤝 贡献
欢迎提交Issue和Pull Request！
作者：谭鸿枭
邮箱：2024302131175@whu.edu.cn
学校：武汉大学
