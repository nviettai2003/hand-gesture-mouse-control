#include "anchors.h"
#include <cmath>
#include <algorithm>

std::vector<Anchor> s_anchors;


struct SsdAnchorsCalculatorOptions {
    int input_size_width, input_size_height;
    float min_scale, max_scale;
    float anchor_offset_x, anchor_offset_y;
    std::vector<int> strides;
    std::vector<float> aspect_ratios;
    bool reduce_boxes_in_lowest_layer;
    float interpolated_scale_aspect_ratio;
    bool fixed_anchor_size;
};

static float CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides) {
    return min_scale + (max_scale - min_scale) * 1.0f * stride_index / (num_strides - 1.0f);
}

static void GenerateAnchors(std::vector<Anchor>& anchors, const SsdAnchorsCalculatorOptions& options) {
    int layer_id = 0;
    while (layer_id < (int)options.strides.size()) {
        std::vector<float> anchor_height, anchor_width, aspect_ratios, scales;
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < (int)options.strides.size() && options.strides[last_same_stride_layer] == options.strides[layer_id]) {
            const float scale = CalculateScale(options.min_scale, options.max_scale, last_same_stride_layer, options.strides.size());
            if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer) {
                aspect_ratios.push_back(1.0f);
                aspect_ratios.push_back(2.0f);
                aspect_ratios.push_back(0.5f);
                scales.push_back(0.1f);
                scales.push_back(scale);
                scales.push_back(scale);
            } else {
                for (float ar : options.aspect_ratios) { 
                    aspect_ratios.push_back(ar);
                    scales.push_back(scale); 
                }
                if (options.interpolated_scale_aspect_ratio > 0.0f) {
                    const float scale_next = last_same_stride_layer == (int)options.strides.size() - 1 ?
                        1.0f : CalculateScale(options.min_scale, options.max_scale, last_same_stride_layer + 1, options.strides.size());
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer++;
        }
        for (int i = 0; i < (int)aspect_ratios.size(); ++i) {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }
        const int stride = options.strides[layer_id];
        int feature_map_height = std::ceil(1.0f * options.input_size_height / stride);
        int feature_map_width  = std::ceil(1.0f * options.input_size_width / stride);
        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (int anchor_id = 0; anchor_id < (int)anchor_height.size(); ++anchor_id) {
                    Anchor a;
                    a.x_center = (x + options.anchor_offset_x) / (float)feature_map_width;
                    a.y_center = (y + options.anchor_offset_y) / (float)feature_map_height;
                    if (options.fixed_anchor_size) { 
                        a.w = 1.0f;
                        a.h = 1.0f; 
                    }
                    else { 
                        a.w = anchor_width[anchor_id];
                        a.h = anchor_height[anchor_id]; 
                    }
                    anchors.push_back(a);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
}

void generate_ssd_anchors() {
    SsdAnchorsCalculatorOptions opt;
    opt.min_scale = 0.1484375f; 
    opt.max_scale = 0.75f;
    opt.input_size_height = 192; 
    opt.input_size_width = 192;
    opt.anchor_offset_x = 0.5f; 
    opt.anchor_offset_y = 0.5f;
    opt.strides = {8,16,16,16};
    opt.aspect_ratios = {1.0f};
    opt.reduce_boxes_in_lowest_layer = false;
    opt.interpolated_scale_aspect_ratio = 1.0f;
    opt.fixed_anchor_size = true;
    GenerateAnchors(s_anchors, opt);
}