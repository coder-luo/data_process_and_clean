
#include "classification.hpp"
#include "common.hpp"

int classification::init(const std::string &modelPath){
    mNet = torch::jit::load(modelPath, torch::kCUDA);
    return eReturnStatus::ProcessSuccess;
}

void classification::setMean(const std::string &str){
    std::vector<std::string> sMean;
    splitString(str, sMean, ',');
    for(auto m : sMean){
        mMean.push_back(std::atof(m.c_str()));
    }
}

void classification::setStd(const std::string &str){
    std::vector<std::string> sStd;
    splitString(str, sStd, ',');
    for(auto s : sStd){
        mStd.push_back(std::atof(s.c_str()));
    }
}

int classification::inference(const cv::Mat &img, clsOutput &out){
    if(img.empty()){
        return eReturnStatus::ProcessFailure;
    }
    torch::Tensor outTensor;
    preProcess(img, outTensor);
    torch::Tensor res = mNet->forward({outTensor}).toTensor();
    postProcess(res, out);
    return eReturnStatus::ProcessSuccess;
}

int classification::inferenceVector(const std::vector<cv::Mat> &imgs, std::vector<clsOutput> &outs){
    for(int i = 0; i < imgs.size(); i++){
        clsOutput out;
        inference(imgs[i], out);
        outs.push_back(out);
    }
    return eReturnStatus::ProcessSuccess;
}

int classification::preProcess(const cv::Mat &img, torch::Tensor &outTensor){
    //根据训练方式设置
    cv::Mat img1, imgFloat;
    cvtColor(img, img1, cv::COLOR_BGR2RGB);
    img1.convertTo(imgFloat, CV_32F, 1.0 / 255);
    if(3 == mMean.size()){
        cv::Scalar mean(mMean[0], mMean[1], mMean[2]);
        imgFloat -= mean;
    }
    if(3 == mStd.size()){
        std::vector<cv::Mat> vImgFloat;
        cv::split(imgFloat, vImgFloat);
        vImgFloat[0] /= mStd[0];
        vImgFloat[1] /= mStd[1];
        vImgFloat[2] /= mStd[2];
        cv::merge(vImgFloat, imgFloat);
    }
    torch::Tensor tImg = torch::CPU(torch::kFloat32).tensorFromBlob(imgFloat.data, { 1, imgFloat.rows, imgFloat.cols, 3 });
    tImg = tImg.permute({ 0,3,1,2 });
    outTensor = torch::autograd::make_variable(tImg, false).to(at::kCUDA);
    return eReturnStatus::ProcessSuccess;
}

int classification::postProcess(const torch::Tensor &inputTensor, clsOutput &out){
    torch::Tensor resTmp = inputTensor.softmax(1).to(torch::kCPU);
    for(int i = 0; i < resTmp.size(1); i++){
        out.softmaxRes.push_back(resTmp[0][i].item().toFloat());
    }
    out.inferenceRes = std::get<1>(resTmp.max(1, true)).item().toInt();
    out.score = out.softmaxRes[out.inferenceRes];
    inputTensor.to(torch::kCPU);
    for(int i = 0; i < inputTensor.size(1); i++){
        out.beforeSoftmaxRes.push_back(inputTensor[0][i].item().toFloat());
    }
    return eReturnStatus::ProcessSuccess;
}