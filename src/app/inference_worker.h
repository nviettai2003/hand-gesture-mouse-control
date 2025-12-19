#ifndef INFERENCE_WORKER_H
#define INFERENCE_WORKER_H

#include "../core/types.h"
#include "../core/frame_buffer.h"
#include "../models/palm.h"
#include "../models/hand_landmark.h"
#include "../mouse/mouse_control.h"
#include <atomic>

class InferenceWorker {
public:
    void run(PALM &palm_detector, HandLandmark &landmark_detector, MouseController &mouse, 
             SafeQueue<cv::Mat> &inputQueue, SafeQueue<detection_output_t> &outputQueue, 
             std::atomic<bool> &running, uint32_t width, uint32_t height);
private:
    void processMouseLogic(MouseController &mouse, const hand_landmark_result_t &res, uint32_t width, uint32_t height);
    bool is_clicking_left = false;
    bool is_clicking_right = false;
};

#endif