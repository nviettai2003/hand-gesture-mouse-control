#ifndef ROI_TRACKER_H
#define ROI_TRACKER_H

#include "../core/types.h"

class RoiTracker {
public:
    static void calculateRoiFromLandmarks(const hand_landmark_result_t& res, HandRoi& raw_roi, int img_w, int img_h);
};

#endif