#include <iostream>
#include "classification.hpp"
#include "common.hpp"
int testNormal() {
    std::string modelPath = "./model/cls.pt";
    std::string sMean = "0.485,0.456,0.406";
    std::string sStd = "0.229,0.224,0.225";

    classification cls;
    cls.setMean(sMean);
    cls.setStd(sStd);
    cls.init(modelPath);

    std::string imgPath = "./data";
    std::vector<std::string> fileNames;
    getFileNameFromFolder(imgPath, fileNames);
    for(auto name : fileNames){
        cv::Mat img = cv::imread(name);
        classification::clsOutput out;
        cls.inference(img, out);
        softmax(out.beforeSoftmaxRes);
    }
    return 0;
}

enum eTestMode{
    IteratedData = 0, //迭代数据
    CleanData    = 1  //清洗数据
};

struct inputParam{
    eTestMode   mode             = CleanData;
    std::string modelProjectPath = "./model/cls1.pt";
    std::string modelProjectMean = "0.485,0.456,0.406";
    std::string modelProjectStd  = "0.229,0.224,0.225";
    cv::Size    modelProjectSize = cv::Size(112,48);
    std::string modelMerge1Path  = "./model/cls2.pt";
    std::string modelMerge1Mean  = "0.485,0.456,0.406";
    std::string modelMerge1Std   = "0.229,0.224,0.225";
    cv::Size    modelMerge1Size  = cv::Size(112,48);
    std::string modelMerge2Path  = "./model/cls3.pt";
    std::string modelMerge2Mean  = "0.485,0.456,0.406";
    std::string modelMerge2Std   = "0.229,0.224,0.225";
    cv::Size    modelMerge2Size  = cv::Size(112,48);
    std::string srcDataPath      = "./data/src";
    std::string dstDataPath      = "./data/save";
    int         classNum         = 2;
    float       thresh           = 0.8;
};

int inference(const inputParam& input){
    //模型初始化
    classification projectModel, merge1Model, merge2Model;
    if(IteratedData == input.mode){
        projectModel.init(input.modelProjectPath);
        projectModel.setMean(input.modelProjectMean);
        projectModel.setMean(input.modelProjectStd);
    }
    merge1Model.init(input.modelMerge1Path);
    merge1Model.setMean(input.modelMerge1Mean);
    merge1Model.setMean(input.modelMerge1Std);
    merge2Model.init(input.modelMerge2Path);
    merge2Model.setMean(input.modelMerge2Mean);
    merge2Model.setMean(input.modelMerge2Std);

    // 创建文件夹
    for(int i = 0; i < input.classNum; i++){
        makeDir(input.dstDataPath + "/" + std::to_string(i) + "N");
        for(int j = 0; j < input.classNum; j++){
            makeDir(input.dstDataPath + "/" + std::to_string(i) + "/" + std::to_string(j));
            makeDir(input.dstDataPath + "/" + std::to_string(i) + "/" + std::to_string(j) + "N");
        }
    }

    //数据处理
    std::vector<std::string> imgPaths;
    getFileNameFromFolder(input.srcDataPath, imgPaths);
    for(int i = 0; i < imgPaths.size(); i++){
        if(0 == i%100){
            std::cout << "processed " << i << " imgs..." << std::endl;
        }
        cv::Mat img = cv::imread(imgPaths[i], 1);
        if(img.empty()) continue;
        classification::clsOutput projectClsOutput;
        std::vector<classification::clsOutput> merge1Outputs, merge2Outputs;
        if(IteratedData == input.mode){
            cv::Mat imgProject;
            cv::resize(img, imgProject, input.modelProjectSize);
            projectModel.inference(img,projectClsOutput);
        }else{
            std::vector<std::string> pathSplit;
            splitString(imgPaths[i], pathSplit, '/');
            projectClsOutput.inferenceRes = std::atoi(pathSplit[pathSplit.size()-2].c_str());
            projectClsOutput.score = 1.0;
        }
        std::vector<cv::Mat> imgs;
        tenCrop(img, imgs, 0.9);
//        for(auto m : imgs){
//            cv::imshow("show", m);
//            cv::waitKey();
//        }
        merge1Model.inferenceVector(imgs, merge1Outputs);
        merge2Model.inferenceVector(imgs, merge2Outputs);
        std::vector<float> mergeFeature(input.classNum);
        for(int m = 0; m < input.classNum; m++){
            for(int n = 0; n < merge1Outputs.size(); n++){
                mergeFeature[m] += merge1Outputs[n].beforeSoftmaxRes[m];
                mergeFeature[m] += merge2Outputs[n].beforeSoftmaxRes[m];
            }
            mergeFeature[m] /= 2*merge1Outputs.size();
        }
        softmax(mergeFeature);
        int maxPosition = max_element(mergeFeature.begin(),mergeFeature.end()) - mergeFeature.begin();
        if(projectClsOutput.score < input.thresh){
            std::string dstPath = input.dstDataPath + "/" + std::to_string(projectClsOutput.inferenceRes) + "N";
            std::string command = "cp " + imgPaths[i] + " " + dstPath;
            system(command.c_str());
        } else {
            if(mergeFeature[maxPosition] < input.thresh){
                std::string dstPath = input.dstDataPath + "/" + std::to_string(projectClsOutput.inferenceRes)
                                      + "/" + std::to_string(maxPosition) + "N";
                std::string command = "cp " + imgPaths[i] + " " + dstPath;
                system(command.c_str());
            }else{
                std::string dstPath = input.dstDataPath + "/" + std::to_string(projectClsOutput.inferenceRes)
                                      + "/" + std::to_string(maxPosition);
                std::string command = "cp " + imgPaths[i] + " " + dstPath;
                system(command.c_str());
            }
        }
    }
    return 1;
}

int main(){
    inputParam input;
    inference(input);
}
