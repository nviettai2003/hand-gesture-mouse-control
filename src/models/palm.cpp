#include "palm.h"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/kernels/register.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

PALM::PALM() {}

void PALM::loadModel(const std::string &palm_model_path) {
    _palm_model = tflite::FlatBufferModel::BuildFromFile(palm_model_path.c_str(), &_palm_error_reporter);
    if (!_palm_model) throw std::runtime_error("Failed to load palm model");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_palm_model.get(), resolver)(&_palm_interpreter);
    if (!_palm_interpreter) throw std::runtime_error("Failed to create palm interpreter");
    _palm_interpreter->SetNumThreads(nthreads);
    if (_palm_interpreter->AllocateTensors() != kTfLiteOk) throw std::runtime_error("Failed to allocate palm tensors");

    _palm_input = _palm_interpreter->inputs()[0];
    TfLiteIntArray *dims = _palm_interpreter->tensor(_palm_input)->dims;
    _palm_in_height = dims->data[1];
    _palm_in_width  = dims->data[2];

    _pPalmInputLayer = _palm_interpreter->typed_tensor<float>(_palm_input);
    _pPalmOutputLayerBbox = _palm_interpreter->typed_tensor<float>(_palm_interpreter->outputs()[0]);
    _pPalmOutputLayerProb = _palm_interpreter->typed_tensor<float>(_palm_interpreter->outputs()[1]);

    generate_ssd_anchors();
}

void PALM::run(const cv::Mat &normalizedImg, palm_detection_result_t &palm_result) {
    palm_result.num = 0;
    if (normalizedImg.empty()) return;
    cv::Mat palmInputMat(_palm_in_height, _palm_in_width, CV_32FC3, (void*)_pPalmInputLayer);
    cv::resize(normalizedImg, palmInputMat, cv::Size(_palm_in_width, _palm_in_height));

    if (_palm_interpreter->Invoke() != kTfLiteOk) return;

    std::list<palm_t> candidates;
    decode_keypoints(candidates, confThreshold);
    std::list<palm_t> final_list;
    non_max_suppression(candidates, final_list, nmsThreshold);
    pack_palm_result(&palm_result, final_list);
}


int PALM::decode_keypoints(std::list<palm_t> &palm_list, float score_thresh) {
    int i = 0;
    for (const auto& anchor : s_anchors) {
        float score0 = _pPalmOutputLayerProb[i];
        float score = 1.0f / (1.0f + std::exp(-score0));
        if (score > score_thresh) {
            float *p = _pPalmOutputLayerBbox + (i * 18);
            float sx = p[0], sy = p[1], w = p[2], h = p[3];
            float cx = sx + anchor.x_center * _palm_in_width;
            float cy = sy + anchor.y_center * _palm_in_height;
            cx /= (float)_palm_in_width; cy /= (float)_palm_in_height;
            w /= (float)_palm_in_width; h /= (float)_palm_in_height;
            palm_t item; item.score = score;
            item.rect.topleft.x = cx - w * 0.5f; item.rect.topleft.y = cy - h * 0.5f;
            item.rect.btmright.x = cx + w * 0.5f; item.rect.btmright.y = cy + h * 0.5f;
            for (int j = 0; j < 7; ++j) {
                float lx = p[4 + 2*j], ly = p[4 + 2*j + 1];
                lx += anchor.x_center * _palm_in_width; ly += anchor.y_center * _palm_in_height;
                item.keys[j].x = lx / (float)_palm_in_width; item.keys[j].y = ly / (float)_palm_in_height;
            }
            palm_list.push_back(item);
        }
        ++i;
    }
    return 0;
}
float PALM::calc_intersection_over_union(rect_t &r0, rect_t &r1) {
    float xmin0 = std::min(r0.topleft.x, r0.btmright.x); float ymin0 = std::min(r0.topleft.y, r0.btmright.y);
    float xmax0 = std::max(r0.topleft.x, r0.btmright.x); float ymax0 = std::max(r0.topleft.y, r0.btmright.y);
    float xmin1 = std::min(r1.topleft.x, r1.btmright.x); float ymin1 = std::min(r1.topleft.y, r1.btmright.y);
    float xmax1 = std::max(r1.topleft.x, r1.btmright.x); float ymax1 = std::max(r1.topleft.y, r1.btmright.y);
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0) return 0.0f;
    float ix = std::max(0.0f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1));
    float iy = std::max(0.0f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1));
    float iarea = ix * iy;
    return iarea / (area0 + area1 - iarea);
}
bool PALM::compare_score(palm_t &a, palm_t &b) { return a.score > b.score; }
int PALM::non_max_suppression(std::list<palm_t> &in, std::list<palm_t> &out, float iou_thresh) {
    in.sort(compare_score);
    for (auto &cand : in) {
        bool ignore = false;
        for (auto &sel : out) {
            if (calc_intersection_over_union(cand.rect, sel.rect) >= iou_thresh) { ignore = true; break; }
        }
        if (!ignore) { out.push_back(cand); if (out.size() >= MAX_PALM_NUM) break; }
    }
    return 0;
}
float PALM::normalize_radians(float angle) { return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI)); }
void PALM::rot_vec(fvec2 &vec, float rotation) {
    float sx = vec.x, sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}
void PALM::compute_rotation(palm_t &palm) {
    float x0 = palm.keys[0].x, y0 = palm.keys[0].y;
    float x2 = palm.keys[2].x, y2 = palm.keys[2].y;
    float target = M_PI * 0.5f;
    palm.rotation = normalize_radians(target - std::atan2(-(y2 - y0), x2 - x0));
}
void PALM::compute_hand_rect(palm_t &palm) {
    float w = palm.rect.btmright.x - palm.rect.topleft.x;
    float h = palm.rect.btmright.y - palm.rect.topleft.y;
    float cx = palm.rect.topleft.x + w * 0.5f;
    float cy = palm.rect.topleft.y + h * 0.5f;
    float shift_x = 0.0f, shift_y = -0.5f;
    float dx = (w * shift_x) * std::cos(palm.rotation) - (h * shift_y) * std::sin(palm.rotation);
    float dy = (w * shift_x) * std::sin(palm.rotation) + (h * shift_y) * std::cos(palm.rotation);
    palm.hand_cx = cx + dx; palm.hand_cy = cy + dy;
    float long_side = std::max(w, h);
    palm.hand_w = long_side * 2.6f; palm.hand_h = long_side * 2.6f;
    float corners[4][2] = {{-palm.hand_w/2, -palm.hand_h/2}, {palm.hand_w/2, -palm.hand_h/2}, 
                           {palm.hand_w/2, palm.hand_h/2}, {-palm.hand_w/2, palm.hand_h/2}};
    for (int i=0; i<4; ++i) {
        fvec2 v{corners[i][0], corners[i][1]}; rot_vec(v, palm.rotation);
        palm.hand_pos[i].x = v.x + palm.hand_cx; palm.hand_pos[i].y = v.y + palm.hand_cy;
    }
}
void PALM::pack_palm_result(palm_detection_result_t *res, std::list<palm_t> &list) {
    int n = 0;
    for (auto &p : list) {
        compute_rotation(p); compute_hand_rect(p);
        if (n < MAX_PALM_NUM) res->palms[n++] = p;
    }
    res->num = n;
}