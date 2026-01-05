/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef RUBIK_CORE_H
#define RUBIK_CORE_H

#include <tensorflow/lite/c/c_api.h>
#include <vector>

// Forward declarations
namespace cv {
class Mat;
}

typedef struct _BOX_RECT {
  int left;
  int right;
  int top;
  int bottom;
} BOX_RECT;

typedef struct RubikDetector {
  TfLiteInterpreter *interpreter;
  TfLiteDelegate *delegate;
  TfLiteModel *model;
} RubikDetector;

typedef struct __detect_result_t {
  int id;
  BOX_RECT box;
  float obj_conf;
} detect_result_t;

// Core functions that can be used by both JNI and Python bindings
RubikDetector* rubik_create(const char* model_path, char** error_msg);
void rubik_destroy(RubikDetector* detector);
std::vector<detect_result_t> rubik_detect(RubikDetector* detector, cv::Mat* input_img, 
                                           double boxThresh, double nmsThreshold, 
                                           char** error_msg);
bool rubik_is_quantized(RubikDetector* detector, char** error_msg);

#endif // RUBIK_CORE_H
