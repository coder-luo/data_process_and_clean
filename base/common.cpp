
#include "common.hpp"
#include <sstream>
#include <dirent.h>
#include <string>
#include "torch/script.h"
#include <unistd.h>

int getFileNameFromFolder(const std::string &folderPath, std::vector<std::string> &fileNames){
    DIR *pDir;
    struct dirent* pDirent;
    pDir = opendir(folderPath.c_str());
    if(NULL == pDir){
        std::cout << "dir is null..." << std::endl;
        return 0;
    }
    while(NULL != (pDirent = readdir(pDir))){
        if(0 == strcmp(pDirent->d_name,".") || 0 == strcmp(pDirent->d_name,"..")){
            continue;
        }
        else if (4 == pDirent->d_type){
            std::string folderPathNew = folderPath + "/" + pDirent->d_name;
            getFileNameFromFolder(folderPathNew, fileNames);
        }
        else{
            std::string fileName = folderPath + "/" + pDirent->d_name;
            fileNames.push_back(fileName);
        }
    }
    closedir(pDir);
    return 1;
}

void splitString(const std::string &str, std::vector<std::string> &outStr, const char div){
    outStr.clear();
    std::istringstream iss(str);
    std::string tmp;
    while (std::getline(iss, tmp, div)) {
        if (tmp != "") {
            outStr.emplace_back(std::move(tmp));
        }
    }
}

void softmax(std::vector<float>& values){
    torch::Tensor tmpValues = torch::tensor(values);
    tmpValues = tmpValues.softmax(0);
    for(int i = 0; i < values.size(); i++){
        values[i] = tmpValues[i].item().toFloat();
    }
}

int tenCrop(cv::Mat &img, std::vector<cv::Mat> &augImgs, const float &ratio){
    if(img.empty()){
        return 0;
    }
    int newWidth = int(ratio*img.cols);
    int newHeight = int(ratio*img.rows);
    int shiftX = int((1-ratio)*img.cols);
    int shiftY = int((1-ratio)*img.rows);
    cv::Mat imgCenter = img(cv::Rect(int(shiftX/2),int(shiftY/2),newWidth, newHeight)).clone() ;
    augImgs.push_back(imgCenter);
    cv::Mat imgCenterFlip;
    cv::flip(imgCenter,imgCenterFlip,1);
    augImgs.push_back(imgCenterFlip);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            cv::Rect rt = cv::Rect(i*shiftX, j*shiftY, newWidth, newHeight);
            cv::Mat imgCenter_;
            imgCenter_ = img(rt).clone();
            augImgs.push_back(imgCenter_);
            cv::Mat imgCenterFlip_;
            cv::flip(imgCenter_,imgCenterFlip_,1);
            augImgs.push_back(imgCenterFlip_);
        }
    }
    return 1;
}

void makeDir(const std::string &path){
    std::vector<std::string> pathSplit;
    splitString(path, pathSplit, '/');
    std::string filePath = "/";
    for(int i = 0; i < pathSplit.size(); i++){
        filePath += pathSplit[i];
        if(!access(filePath.c_str(),0)){
            filePath += "/";
            continue;
        }
        std::string command = "mkdir " + filePath;
        system(command.c_str());
        filePath += "/";
    }
}