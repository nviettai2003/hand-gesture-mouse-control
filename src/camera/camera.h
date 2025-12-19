#ifndef CAMERA_H
#define CAMERA_H

#include "../core/types.h" 
#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/formats.h>
#include <memory>
#include <queue>
#include <mutex>
#include <map>

using namespace libcamera;

class SimpleCamera {
public:
    SimpleCamera();
    ~SimpleCamera();
    bool initCamera();
    void configureStill(uint32_t width, uint32_t height);
    bool startCamera();
    bool readFrame(LibcameraOutData &out);
    void returnFrameBuffer(LibcameraOutData &frameData);
    void stopCamera();
    void closeCamera();

private:
    void requestComplete(Request *request);
    std::unique_ptr<CameraManager> cm;
    std::shared_ptr<Camera> camera_;
    std::unique_ptr<CameraConfiguration> config_;
    std::unique_ptr<FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<Request>> requests_;
    std::map<int, std::pair<void*, unsigned int>> mappedBuffers_;
    std::queue<Request*> requestQueue;
    std::mutex queue_mutex_;
    bool camera_acquired_ = false;
    bool camera_started_ = false;
};
#endif
