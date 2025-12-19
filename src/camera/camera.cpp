#include "camera.h"
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <libcamera/control_ids.h>

SimpleCamera::SimpleCamera() {}
SimpleCamera::~SimpleCamera() { closeCamera(); }

bool SimpleCamera::initCamera() {
    cm = std::make_unique<CameraManager>();
    if (cm->start()) { std::cerr << "Failed to start CameraManager\n"; return false; }
    if (cm->cameras().empty()) { std::cerr << "No cameras\n"; return false; }
    camera_ = cm->get(cm->cameras()[0]->id());
    if (!camera_) return false;
    if (camera_->acquire()) return false;
    camera_acquired_ = true;
    return true;
}

void SimpleCamera::configureStill(uint32_t width, uint32_t height) {
    config_ = camera_->generateConfiguration({ StreamRole::VideoRecording });
    if (width && height) config_->at(0).size = libcamera::Size(width, height);
    config_->at(0).pixelFormat = formats::RGB888;
    if (config_->validate() == CameraConfiguration::Invalid) throw std::runtime_error("Invalid config");
}

bool SimpleCamera::startCamera() {
    if (camera_->configure(config_.get()) < 0) return false;
    camera_->requestCompleted.connect(this, &SimpleCamera::requestComplete);
    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    for (auto &cfg : *config_) {
        if (allocator_->allocate(cfg.stream()) < 0) return false;
    }
    for (size_t i = 0; i < allocator_->buffers(config_->at(0).stream()).size(); ++i) {
        auto req = camera_->createRequest();
        if (!req) return false;
        auto &buffers = allocator_->buffers(config_->at(0).stream());
        req->addBuffer(config_->at(0).stream(), buffers[i].get());
        for (auto &plane : buffers[i]->planes()) {
            void *mem = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
            mappedBuffers_[plane.fd.get()] = {mem, plane.length};
        }
        requests_.push_back(std::move(req));
    }

    ControlList controls(camera_->controls());
    int64_t frame_time = 1000000 / 30;
    controls.set(controls::FrameDurationLimits, {frame_time, frame_time});

    if (camera_->start(&controls)) return false;
    camera_started_ = true;
    for (auto &req : requests_) camera_->queueRequest(req.get());
    return true;
}

bool SimpleCamera::readFrame(LibcameraOutData &out) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (requestQueue.empty()) return false;
    Request *req = requestQueue.front();
    auto &buffers = req->buffers();
    for (auto &it : buffers) {
        FrameBuffer *buffer = it.second; // Đây là libcamera::FrameBuffer (OK)
        auto &plane = buffer->planes()[0];
        out.imageData = (uint8_t*)mappedBuffers_[plane.fd.get()].first;
        out.size = plane.length;
    }
    out.request = (uint64_t)req;
    requestQueue.pop();
    return true;
}

void SimpleCamera::returnFrameBuffer(LibcameraOutData &frameData) {
    Request *req = (Request*)frameData.request;
    req->reuse(Request::ReuseBuffers);
    camera_->queueRequest(req);
}

void SimpleCamera::stopCamera() {
    if (camera_ && camera_started_) camera_->stop();
    camera_started_ = false;
}

void SimpleCamera::closeCamera() {
    stopCamera();
    for (auto &m : mappedBuffers_) {
        if (m.second.first) munmap(m.second.first, m.second.second);
    }
    mappedBuffers_.clear();
    if (camera_acquired_) camera_->release();
    camera_acquired_ = false;
}

void SimpleCamera::requestComplete(Request *request) {
    if (request->status() != Request::RequestCancelled) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        requestQueue.push(request);
    }
}