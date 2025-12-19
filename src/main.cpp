#include <iostream>
#include <thread>
#include <atomic>

#include "core/app_config.h"
#include "core/frame_buffer.h" 
#include "camera/camera.h"
#include "models/palm.h"
#include "models/hand_landmark.h"
#include "mouse/mouse_control.h"

#include "app/capture_worker.h"
#include "app/inference_worker.h"
#include "app/renderer.h"

int main() {
    uint32_t width = 800;
    uint32_t height = 600;

    SimpleCamera cam;
    if (!cam.initCamera()) return -1;
    cam.configureStill(width, height);

    PALM palmDetector;
    HandLandmark handDetector;
    try {
        palmDetector.loadModel(PALM_MODEL_PATH);
        handDetector.loadModel(HAND_LANDMARK_MODEL_PATH);
    } catch (const std::exception &e) {
        std::cerr << "Model Error: " << e.what() << std::endl;
        return -1;
    }

    MouseController mouse;
    if (!mouse.init()) {
        std::cerr << "WARNING: Mouse init failed. Run with sudo?\n";
    }

    if (!cam.startCamera()) return -1;

    SafeQueue<cv::Mat> capBuf(2);
    SafeQueue<detection_output_t> outBuf(2);
    std::atomic<bool> running{true};

    CaptureWorker capWorker;
    InferenceWorker inferWorker;
    Renderer renderer;

    std::thread t1(&CaptureWorker::run, &capWorker, std::ref(cam), std::ref(capBuf), std::ref(running), width, height);
    
    std::thread t2(&InferenceWorker::run, &inferWorker, 
                   std::ref(palmDetector), std::ref(handDetector), std::ref(mouse),
                   std::ref(capBuf), std::ref(outBuf), 
                   std::ref(running), width, height);
    
    std::thread t3(&Renderer::run, &renderer, std::ref(outBuf), std::ref(running), width, height);

    t1.join();
    t2.join();
    t3.join();

    cam.stopCamera();
    capBuf.stop();
    outBuf.stop();

    return 0;
}
