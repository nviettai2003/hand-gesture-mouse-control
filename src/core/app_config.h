#ifndef APP_CONFIG_H
#define APP_CONFIG_H

// Model Paths
#define PALM_MODEL_PATH "./models/palm_detection_lite.tflite"
#define HAND_LANDMARK_MODEL_PATH "./models/hand_landmark_lite.tflite"

// Constants
#define MAX_PALM_NUM 4
#define HAND_JOINT_NUM 21

// Screen & Mouse Config
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080
#define MOUSE_REGION_W 560
#define MOUSE_REGION_H 315

// Thresholds
#define THRESH_TRACK_ENTER 0.5f
#define THRESH_TRACK_EXIT  0.4f

#endif