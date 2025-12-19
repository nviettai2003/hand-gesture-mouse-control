#ifndef TYPES_H
#define TYPES_H

#include <opencv2/core.hpp>
#include <vector>
#include <stdint.h>
#include "app_config.h"

// Basic Math Types
struct fvec2 { float x, y; };
struct fvec3 { float x, y, z; };
struct rect_t { fvec2 topleft; fvec2 btmright; };

// Libcamera Wrapper Data
struct LibcameraOutData {
    uint8_t *imageData;
    uint32_t size;
    uint64_t request;
};

// Palm Detection Structures
struct palm_t {
    float hand_cx, hand_cy, hand_w, hand_h; // Normalized
    fvec2 keys[7];
    float score;
    rect_t rect;
    float rotation; // Radian
    fvec2 hand_pos[4]; 
};

struct palm_detection_result_t {
    int num;
    palm_t palms[MAX_PALM_NUM];
};

// Hand Landmark Structures
struct hand_landmark_result_t {
    float score;
    fvec3 joint[HAND_JOINT_NUM]; // Pixel coords
    int frame_width;
    int frame_height;
};

// ROI Tracking Structure
struct HandRoi {
    float xc, yc, w, h; // Normalized
    float rotation;     // Radian
    bool isValid = false; 
};

// Output Data for Renderer
struct detection_output_t {
    cv::Mat frame;
    std::vector<hand_landmark_result_t> hand_results;
    bool is_tracking;
    double palm_time_ms;
    double hand_time_ms;
};

#endif