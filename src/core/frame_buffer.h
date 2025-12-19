#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

template<typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;
    std::atomic<bool> stopped;

public:
    SafeQueue(size_t cap) : max_size(cap), stopped(false) {}

    void push(T item) {
        std::unique_lock<std::mutex> lk(mtx);
        if (stopped) return;
        if (q.size() >= max_size) q.pop();
        q.push(std::move(item));
        cv.notify_one();
    }

    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this]{ return !q.empty() || stopped; });
        if (stopped && q.empty()) return false;
        out = std::move(q.front());
        q.pop();
        return true;
    }

    void stop() {
        stopped = true;
        cv.notify_all();
    }
};

#endif
