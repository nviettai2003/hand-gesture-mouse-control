#include "capture_worker.h"
#include <opencv2/imgproc.hpp>
#include <thread>
#include <iostream>

void CaptureWorker::run(SimpleCamera &cam, SafeQueue<cv::Mat> &frameQueue, std::atomic<bool> &running, uint32_t width, uint32_t height) {
    while (running.load()) {
        LibcameraOutData fd;
        if (!cam.readFrame(fd)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        cv::Mat rawData((int)height, (int)width, CV_8UC3, fd.imageData);
        cv::Mat safeFrame = rawData.clone();
        cam.returnFrameBuffer(fd);

        cv::flip(safeFrame, safeFrame, 1);
        frameQueue.push(std::move(safeFrame));
    }
}