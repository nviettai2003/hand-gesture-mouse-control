#ifndef RENDERER_H
#define RENDERER_H

#include "../core/types.h"
#include "../core/frame_buffer.h"
#include <atomic>

class Renderer {
public:
    void run(SafeQueue<detection_output_t> &outputQueue, std::atomic<bool> &running, uint32_t width, uint32_t height);
};

#endif