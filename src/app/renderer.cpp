#include "renderer.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

void Renderer::run(SafeQueue<detection_output_t> &outputQueue, std::atomic<bool> &running, uint32_t width, uint32_t height) {
    cv::namedWindow("Hand Tracking Final", cv::WINDOW_FULLSCREEN);
    int reg_x = (width - MOUSE_REGION_W) / 2;
    int reg_y = (height - MOUSE_REGION_H) / 2;
    cv::Rect mouse_rect(reg_x, reg_y, MOUSE_REGION_W, MOUSE_REGION_H);

    double fps = 0.0;
    int frame_counter = 0;
    auto last_fps_time = std::chrono::high_resolution_clock::now();

    detection_output_t out;
    while (running.load()) {
        if (!outputQueue.pop(out)) break;
        if (out.frame.empty()) continue;

        frame_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(current_time - last_fps_time).count();
        if (elapsed_sec >= 1.0) { 
            fps = frame_counter / elapsed_sec;
            frame_counter = 0;
            last_fps_time = current_time;
        }

        cv::rectangle(out.frame, mouse_rect, cv::Scalar(0, 255, 255), 2);
        
        for (const auto &h : out.hand_results) {
             const std::vector<std::pair<int, int>> connections = {
                {0,1}, {1,2}, {2,3}, {3,4}, {0,5}, {5,6}, {6,7}, {7,8},
                {5,9}, {9,10}, {10,11}, {11,12}, {9,13}, {13,14}, {14,15}, {15,16},
                {13,17}, {0,17}, {17,18}, {18,19}, {19,20}
            };
            for (auto& c : connections) {
                cv::line(out.frame, cv::Point(h.joint[c.first].x, h.joint[c.first].y),
                         cv::Point(h.joint[c.second].x, h.joint[c.second].y), cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            }
            cv::circle(out.frame, cv::Point(h.joint[9].x, h.joint[9].y), 6, cv::Scalar(0,0,255), -1);
            for (int i = 0; i < HAND_JOINT_NUM; i++) {
                if(i != 9) cv::circle(out.frame, cv::Point(h.joint[i].x, h.joint[i].y), 4, (i==0?cv::Scalar(0,0,255):cv::Scalar(0,255,0)), -1, cv::LINE_AA);
            }
        }

        cv::putText(out.frame, out.is_tracking ? "Tracking" : "Searching", cv::Point(10, 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, out.is_tracking ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
        
        std::stringstream ss_palm; ss_palm << "Palm: " << std::fixed << std::setprecision(1) << out.palm_time_ms << "ms";
        cv::putText(out.frame, ss_palm.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);

        std::stringstream ss_hand; ss_hand << "Hand: " << std::fixed << std::setprecision(1) << out.hand_time_ms << "ms";
        cv::putText(out.frame, ss_hand.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);

        std::stringstream ss_fps; ss_fps << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(out.frame, ss_fps.str(), cv::Point(out.frame.cols - 130, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Hand Tracking Final", out.frame);
        if (cv::waitKey(1) == 27) running.store(false);
    }
}