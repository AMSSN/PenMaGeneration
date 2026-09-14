cmake_minimum_required(VERSION 3.16)
project(PenMaGeneration VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_USE_MATH_DEFINES)

if(MSVC)
    add_compile_options("/source-charset:utf-8")
endif()

# Halcon
set(HALCON_DIR ${CMAKE_SOURCE_DIR}/3rdpartys/halcon)
include_directories(${HALCON_DIR}/include)
include_directories(${HALCON_DIR}/include/halconcpp)
link_directories(${HALCON_DIR}/lib/x64-win64)
link_libraries(halconcpp)

# OpenCV
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/3rdpartys/opencv)
include_directories(${OPENCV_DIR}/include)
include_directories(${OPENCV_DIR}/include/opencv2)
link_directories(${OPENCV_DIR}/lib/x64/vc15)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    link_libraries(opencv_world452d)
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
    link_libraries(opencv_world452)
endif()


#导入spdlog
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdpartys/spdlog)
message(STATUS "The CMAKE_CURRENT_SOURCE_DIR is: ${CMAKE_CURRENT_SOURCE_DIR}")

# 自动收集 src 目录下所有 .cc 和 .cpp 文件
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cc
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE LIB_HEADERS
    ${CMAKE_SOURCE_DIR}/src/*.hh
    ${CMAKE_SOURCE_DIR}/src/*.h
)

set(PROJECT_SOURCES
    main.cpp
    ${LIB_SOURCES}        # ← 现在是实际的 .cc 文件列表
    ${LIB_HEADERS}        # ← 加上头文件，方便 IDE 显示
)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

# 让编译器能找到 src 下的头文件
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
