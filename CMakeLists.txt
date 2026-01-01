cmake_minimum_required(VERSION 3.10)
project(GLCM_WaterForest_Extraction VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找OpenMP（可选，用于并行加速）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found, parallel processing enabled")
endif()

# 设置源文件
set(SOURCES
    main.cpp
    GLCMExtractor.cpp
    GUIInterface.cpp
)

# 设置头文件
set(HEADERS
    GLCMExtractor.h
    GUIInterface.h
)

# 创建可执行文件
add_executable(glcm_extractor ${SOURCES} ${HEADERS})

# 链接OpenCV库
target_link_libraries(glcm_extractor ${OpenCV_LIBS})

# 如果找到OpenMP，链接它
if(OpenMP_CXX_FOUND)
    target_link_libraries(glcm_extractor OpenMP::OpenMP_CXX)
endif()

# 设置输出目录
set_target_properties(glcm_extractor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制示例图像到输出目录（如果存在）
file(GLOB BMP_FILES "${CMAKE_SOURCE_DIR}/*.bmp")
if(BMP_FILES)
    foreach(BMP_FILE ${BMP_FILES})
        get_filename_component(BMP_NAME ${BMP_FILE} NAME)
        configure_file(${BMP_FILE} ${CMAKE_BINARY_DIR}/bin/${BMP_NAME} COPYONLY)
    endforeach()
endif()

# 打印配置信息
message(STATUS "==========================================")
message(STATUS "  GLCM Water/Forest Extraction System")
message(STATUS "==========================================")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==========================================")

# 安装目标
install(TARGETS glcm_extractor DESTINATION bin)
