#ifndef ANCHORS_H
#define ANCHORS_H
#include <vector>
#include "../core/types.h"


typedef struct Anchor { float x_center, y_center, w, h; } Anchor;

extern std::vector<Anchor> s_anchors;
void generate_ssd_anchors();
#endif