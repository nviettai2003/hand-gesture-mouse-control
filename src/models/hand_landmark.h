#ifndef HAND_LANDMARK_H
#define HAND_LANDMARK_H

#include "../core/types.h"
#include <string>
#include <memory>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/stderr_reporter.h>

class HandLandmark {
public:
    HandLandmark() {}
    void loadModel(const std::string &model_path);
    void run(const cv::Mat &frame_bgr, 
             std::vector<hand_landmark_result_t> &hand_results, 
             const HandRoi &roi, 
             int img_width, int img_height);
    float confThreshold = 0.5f;
    int nthreads = 3;

private:
    std::unique_ptr<tflite::FlatBufferModel> _hand_model;
    std::unique_ptr<tflite::Interpreter> _hand_interpreter;
    tflite::StderrReporter _hand_error_reporter;
    int _hand_input = -1;
    float *_pHandInputLayer = nullptr;
    float *_pHandOutputLayerLandmarks = nullptr;
    float *_pHandOutputLayerScore = nullptr;
    int _hand_in_width = 224;
    int _hand_in_height = 224;
};
#endif