
#ifndef MAIN_CLASSIFICATION_HPP
#define MAIN_CLASSIFICATION_HPP

#include "torch/script.h"
#include "opencv2/opencv.hpp"

class classification {
public:
    typedef struct classificationResult{
        std::vector<float> softmaxRes; // 分类结果, softmax后概率
        std::vector<float> beforeSoftmaxRes; // 分类结果, softmax前特征
        int inferenceRes = -1; // 类别结果
        float score = 0.0; // 当前类别对应的得分
    }clsOutput;
public:
    classification(){;}
    ~classification(){;}

    ///////////////////////////////////
    //函数功能：模型初始化
    //输入：modelPath　模型地址
    ///////////////////////////////////
    int init(const std::string &modelPath);

    ///////////////////////////////////
    //函数功能：模型推理，支持单张图
    //输入：img 输入图片
    //输出：out 推理结果
    ///////////////////////////////////
    int inference(const cv::Mat &img, clsOutput &out);

    ///////////////////////////////////
    //函数功能：模型推理，支持多张图
    //输入：imgs 输入图片组
    //输出：outs 推理结果
    ///////////////////////////////////
    int inferenceVector(const std::vector<cv::Mat> &imgs, std::vector<clsOutput> &outs);

    ///////////////////////////////////
    //函数功能：设置模型均值参数
    //输入：str 均值参数
    ///////////////////////////////////
    void setMean(const std::string &str);

    ///////////////////////////////////
    //函数功能：设置模型方差参数
    //输入：str 方差参数
    ///////////////////////////////////
    void setStd(const std::string &str);

private:
    ///////////////////////////////////
    //函数功能：图片预处理
    //输入：img 输入图片
    //输出：outTensor 预处理后的结果
    ///////////////////////////////////
    int preProcess(const cv::Mat &img, torch::Tensor &outTensor);

    ///////////////////////////////////
    //函数功能：模型推理结果后处理
    //输入：inputTensor 输入推理后的特征
    //输出：out 后处理最终结果
    ///////////////////////////////////
    int postProcess(const torch::Tensor &inputTensor, clsOutput &out);

private:
    std::shared_ptr<torch::jit::script::Module> mNet;
    std::vector<float> mMean;
    std::vector<float> mStd;
};


#endif //MAIN_CLASSIFICATION_HPP
