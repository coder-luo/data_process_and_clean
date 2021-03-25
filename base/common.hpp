
#ifndef MAIN_COMMON_HPP
#define MAIN_COMMON_HPP

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

enum eReturnStatus{
    ProcessSuccess = 0,
    ProcessFailure = 1
};

///////////////////////////////////
//函数功能：获取文件夹下所有文件的名称
//输入：folderPath 文件夹地址
//输出：fileNames 文件名称
///////////////////////////////////
int getFileNameFromFolder(const std::string &folderPath, std::vector<std::string> &fileNames);

///////////////////////////////////
//函数功能：分割字符串
//输入：str 待分割字符串, div 分割符
//输出：outStr 分割结果
///////////////////////////////////
void splitString(const std::string &str, std::vector<std::string> &outStr, const char div = ' ');

///////////////////////////////////
//函数功能：softmax计算
//输入：values
//输出：values
///////////////////////////////////
void softmax(std::vector<float>& values);

///////////////////////////////////
//函数功能：图像扩增方法, ten crop
//输入：img 输入图像,  ratio crop的比例，默认0.9
//输出：augImgs 扩增后图像
///////////////////////////////////
int tenCrop(cv::Mat &img, std::vector<cv::Mat> &augImgs, const float &ratio = 0.9);

///////////////////////////////////
//函数功能：创建文件夹
//输入：path 文件夹地址,支持同时创建多级目录
///////////////////////////////////
void makeDir(const std::string &path);


#endif //MAIN_COMMON_HPP
