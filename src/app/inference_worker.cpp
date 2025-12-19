#include "inference_worker.h"
#include "../tracking/roi_tracker.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>

void InferenceWorker::run(PALM &palm_detector, HandLandmark &landmark_detector, MouseController &mouse, 
             SafeQueue<cv::Mat> &inputQueue, SafeQueue<detection_output_t> &outputQueue, 
             std::atomic<bool> &running, uint32_t width, uint32_t height) 
{
    HandRoi current_roi;
    current_roi.isValid = false;

    cv::Mat frame;
    while (running.load()) {
        if (!inputQueue.pop(frame)) break;
        if (frame.empty()) continue;

        detection_output_t out_data;
        out_data.frame = frame.clone();
        out_data.is_tracking = false;
        out_data.palm_time_ms = 0.0;
        out_data.hand_time_ms = 0.0;

        std::vector<hand_landmark_result_t> hand_results;
        bool hand_found = false;

        // --- 1. TRACKING MODE ---
        if (current_roi.isValid) {
            auto t1 = std::chrono::high_resolution_clock::now();
            landmark_detector.run(frame, hand_results, current_roi, width, height);
            auto t2 = std::chrono::high_resolution_clock::now();
            out_data.hand_time_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (!hand_results.empty()) {
                float score = hand_results[0].score;
                if (score > THRESH_TRACK_EXIT) {
                    hand_found = true;
                    out_data.is_tracking = true;
                    if (score > 0.5f) {
                        HandRoi raw_roi;
                        RoiTracker::calculateRoiFromLandmarks(hand_results[0], raw_roi, width, height);
                        current_roi = raw_roi;
                        processMouseLogic(mouse, hand_results[0], width, height);
                    }
                } else {
                    current_roi.isValid = false;
                }
            } else {
                current_roi.isValid = false;
            }
        }

        // --- 2. DETECTION MODE (PALM) ---
        if (!hand_found) {
            cv::Mat normalizedImg;
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            rgb.convertTo(normalizedImg, CV_32FC3, 1.0f / 255.0f);

            palm_detection_result_t palm_result;
            auto t1 = std::chrono::high_resolution_clock::now();
            palm_detector.run(normalizedImg, palm_result);
            auto t2 = std::chrono::high_resolution_clock::now();
            out_data.palm_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (palm_result.num > 0) {
                const auto& p = palm_result.palms[0];
                HandRoi roi_from_palm;
                roi_from_palm.xc = p.hand_cx; roi_from_palm.yc = p.hand_cy;
                roi_from_palm.w = p.hand_w; roi_from_palm.h = p.hand_h;
                roi_from_palm.rotation = p.rotation; roi_from_palm.isValid = true;

                auto t3 = std::chrono::high_resolution_clock::now();
                landmark_detector.run(frame, hand_results, roi_from_palm, width, height);
                auto t4 = std::chrono::high_resolution_clock::now();
                out_data.hand_time_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();

                if (!hand_results.empty() && hand_results[0].score > THRESH_TRACK_ENTER) {
                    HandRoi raw_roi;
                    RoiTracker::calculateRoiFromLandmarks(hand_results[0], raw_roi, width, height);
                    current_roi = raw_roi;
                }
            }
        }
        out_data.hand_results = hand_results;
        outputQueue.push(std::move(out_data));
    }
}

void InferenceWorker::processMouseLogic(MouseController &mouse, const hand_landmark_result_t &res, uint32_t width, uint32_t height) {
    const float region_w = (float)MOUSE_REGION_W;
    const float region_h = (float)MOUSE_REGION_H;
    const float offset_x = (width - region_w) / 2.0f;
    const float offset_y = (height - region_h) / 2.0f;

    float hx = res.joint[9].x;
    float hy = res.joint[9].y;

    if (hx < offset_x) hx = offset_x;
    if (hx > offset_x + region_w) hx = offset_x + region_w;
    if (hy < offset_y) hy = offset_y;
    if (hy > offset_y + region_h) hy = offset_y + region_h;

    float x_norm = (hx - offset_x) / region_w;
    float y_norm = (hy - offset_y) / region_h;

    int abs_x = (int)(x_norm * SCREEN_WIDTH);
    int abs_y = (int)(y_norm * SCREEN_HEIGHT);
    mouse.move_absolute(abs_x, abs_y);

    // Tính toán ngưỡng click dựa trên kích thước bàn tay
    float scale_dist = std::sqrt(std::pow(res.joint[5].x - res.joint[9].x, 2) + std::pow(res.joint[5].y - res.joint[9].y, 2));
    float click_thresh = scale_dist * 1.3f;

    // Khoảng cách ngón trỏ (8) và ngón giữa (12) -> Click Trái
    float d_left = std::sqrt(std::pow(res.joint[8].x - res.joint[12].x, 2) + std::pow(res.joint[8].y - res.joint[12].y, 2));
    
    // Khoảng cách ngón cái (4) và gốc ngón trỏ (6) -> Click Phải
    float d_right = std::sqrt(std::pow(res.joint[4].x - res.joint[6].x, 2) + std::pow(res.joint[4].y - res.joint[6].y, 2));

    
    if (d_left < click_thresh) {
        if (!is_clicking_left) {
            mouse.press_left();      
            is_clicking_left = true;  
        }
    } 
    else {
        if (is_clicking_left) {
            mouse.release_left();     
            is_clicking_left = false; 
        }
    }

    if (!is_clicking_left && d_right < click_thresh) {
        if (!is_clicking_right) {
            mouse.click_right();
            is_clicking_right = true;
        }
    } else {
        is_clicking_right = false;
    }
}