#include "hand_landmark.h"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/kernels/register.h>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace cv;

void HandLandmark::loadModel(const std::string &path) {
    _hand_model = tflite::FlatBufferModel::BuildFromFile(path.c_str(), &_hand_error_reporter);
    if (!_hand_model) throw std::runtime_error("Failed to load hand model");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_hand_model.get(), resolver)(&_hand_interpreter);
    if (!_hand_interpreter) throw std::runtime_error("Failed to create hand interpreter");

    _hand_interpreter->SetNumThreads(nthreads);
    if (_hand_interpreter->AllocateTensors() != kTfLiteOk) throw std::runtime_error("Failed to allocate hand tensors");

    _hand_input = _hand_interpreter->inputs()[0];
    TfLiteIntArray *dims = _hand_interpreter->tensor(_hand_input)->dims;
    _hand_in_height = dims->data[1];
    _hand_in_width  = dims->data[2];

    _pHandInputLayer = _hand_interpreter->typed_tensor<float>(_hand_input);
    _pHandOutputLayerLandmarks = _hand_interpreter->typed_tensor<float>(_hand_interpreter->outputs()[0]);
    _pHandOutputLayerScore = _hand_interpreter->typed_tensor<float>(_hand_interpreter->outputs()[1]);
}

cv::Mat getHandAffineTransform(const HandRoi &roi, int img_w, int img_h, int target_w, int target_h) {
    float cx = roi.xc * img_w; float cy = roi.yc * img_h;
    float w = roi.w * img_w; float h = roi.h * img_h;
    float rotation = roi.rotation; 
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f(cx, cy);
    float vec_x = 0.0f; float vec_y = -h * 0.5f;
    float rot_vec_x = vec_x * cos(rotation) - vec_y * sin(rotation);
    float rot_vec_y = vec_x * sin(rotation) + vec_y * cos(rotation);
    srcTri[1] = cv::Point2f(cx + rot_vec_x, cy + rot_vec_y);
    vec_x = -w * 0.5f; vec_y = 0.0f;
    rot_vec_x = vec_x * cos(rotation) - vec_y * sin(rotation);
    rot_vec_y = vec_x * sin(rotation) + vec_y * cos(rotation);
    srcTri[2] = cv::Point2f(cx + rot_vec_x, cy + rot_vec_y);
    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f(target_w * 0.5f, target_h * 0.5f);
    dstTri[1] = cv::Point2f(target_w * 0.5f, 0.0f);
    dstTri[2] = cv::Point2f(0.0f, target_h * 0.5f);
    return cv::getAffineTransform(srcTri, dstTri);
}

void HandLandmark::run(const cv::Mat &frame_bgr, std::vector<hand_landmark_result_t> &hand_results, 
                       const HandRoi &roi, int img_width, int img_height) {
    hand_results.clear();
    if (frame_bgr.empty()) return;

    cv::Mat affine = getHandAffineTransform(roi, img_width, img_height, _hand_in_width, _hand_in_height);
    cv::Mat affineInv;
    cv::invertAffineTransform(affine, affineInv);

    cv::Mat crop_bgr;
    cv::warpAffine(frame_bgr, crop_bgr, affine, cv::Size(_hand_in_width, _hand_in_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    
    cv::Mat crop_rgb;
    cv::cvtColor(crop_bgr, crop_rgb, cv::COLOR_BGR2RGB);
    cv::Mat inputTensorMat(_hand_in_height, _hand_in_width, CV_32FC3, _pHandInputLayer);
    crop_rgb.convertTo(inputTensorMat, CV_32FC3, 1.0f / 255.0f);

    if (_hand_interpreter->Invoke() != kTfLiteOk) return;

    float score = _pHandOutputLayerScore[0];
    if (score > 0.1f) { 
        hand_landmark_result_t res;
        res.score = score;
        res.frame_width = img_width; res.frame_height = img_height;
        for (int j = 0; j < HAND_JOINT_NUM; ++j) {
            float x_out = _pHandOutputLayerLandmarks[3 * j + 0];
            float y_out = _pHandOutputLayerLandmarks[3 * j + 1];
            double x_orig = affineInv.at<double>(0, 0) * x_out + affineInv.at<double>(0, 1) * y_out + affineInv.at<double>(0, 2);
            double y_orig = affineInv.at<double>(1, 0) * x_out + affineInv.at<double>(1, 1) * y_out + affineInv.at<double>(1, 2);
            res.joint[j].x = (float)x_orig; res.joint[j].y = (float)y_orig; res.joint[j].z = 0; 
        }
        hand_results.push_back(res);
    }
}