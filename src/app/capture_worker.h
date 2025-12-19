#ifndef CAPTURE_WORKER_H
#define CAPTURE_WORKER_H

#include "../camera/camera.h"
#include "../core/frame_buffer.h" 
#include <opencv2/core.hpp>
#include <atomic>

class CaptureWorker {
public:
    void run(SimpleCamera &cam, SafeQueue<cv::Mat> &frameQueue, std::atomic<bool> &running, uint32_t width, uint32_t height);
};

#endif