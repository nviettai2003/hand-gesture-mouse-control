#ifndef PALM_H
#define PALM_H

#include <list>
#include <string>
#include <memory>
#include "../core/types.h"
#include "anchors.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/stderr_reporter.h>

class PALM {
public:
    PALM();
    void loadModel(const std::string &palm_model_path);
    void run(const cv::Mat &normalizedImg, palm_detection_result_t &palm_result);

    float confThreshold = 0.5f;
    float nmsThreshold = 0.3f;
    int nthreads = 2;

private:
    std::unique_ptr<tflite::FlatBufferModel> _palm_model;
    std::unique_ptr<tflite::Interpreter> _palm_interpreter;
    tflite::StderrReporter _palm_error_reporter;
    int _palm_input = -1;
    float *_pPalmInputLayer = nullptr;
    float *_pPalmOutputLayerBbox = nullptr;
    float *_pPalmOutputLayerProb = nullptr;
    int _palm_in_width = 192;
    int _palm_in_height = 192;

    int decode_keypoints(std::list<palm_t> &palm_list, float score_thresh);
    float calc_intersection_over_union(rect_t &rect0, rect_t &rect1);
    static bool compare_score(palm_t &a, palm_t &b);
    int non_max_suppression(std::list<palm_t> &in, std::list<palm_t> &out, float iou_thresh);
    float normalize_radians(float angle);
    void rot_vec(fvec2 &vec, float rotation);
    void compute_rotation(palm_t &palm);
    void compute_hand_rect(palm_t &palm);
    void pack_palm_result(palm_detection_result_t *palm_result, std::list<palm_t> &palm_list);
};
#endif