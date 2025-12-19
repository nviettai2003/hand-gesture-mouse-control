#include "roi_tracker.h"
#include <cmath>
#include <algorithm>

void RoiTracker::calculateRoiFromLandmarks(const hand_landmark_result_t& res, HandRoi& raw_roi, int img_w, int img_h) {
    float x0 = res.joint[0].x; float y0 = res.joint[0].y;
    float x9 = res.joint[9].x; float y9 = res.joint[9].y;
    float x5 = res.joint[5].x; float y5 = res.joint[5].y;
    float x17 = res.joint[17].x; float y17 = res.joint[17].y;

    float angle = std::atan2(y9 - y0, x9 - x0);
    float rotation = angle - (-M_PI / 2.0f);

    float cx = (x0 + x9) / 2.0f;
    float cy = (y0 + y9) / 2.0f;

    float dist_spine = std::sqrt(std::pow(x9 - x0, 2) + std::pow(y9 - y0, 2));
    float size_spine = dist_spine * 2.6f;
    float dist_palm = std::sqrt(std::pow(x17 - x5, 2) + std::pow(y17 - y5, 2));
    float size_palm = dist_palm * 3.5f; 
    float size = std::max(size_spine, size_palm);

    float shift_x = 0.0f;
    float shift_y = -0.15f;
    float dx = shift_x * std::cos(rotation) - shift_y * std::sin(rotation);
    float dy = shift_x * std::sin(rotation) + shift_y * std::cos(rotation);
    cx += dx * size;
    cy += dy * size;

    raw_roi.xc = cx / img_w;
    raw_roi.yc = cy / img_h;
    raw_roi.w = size / img_w;
    raw_roi.h = size / img_h;
    raw_roi.rotation = rotation;
    raw_roi.isValid = true;
}