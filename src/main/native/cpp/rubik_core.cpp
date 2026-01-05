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

#include "rubik_core.h"
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Debug print macro
#ifndef NDEBUG
#define DEBUG_PRINT(...) std::printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
  } while (0)
#endif

// Helper function for proper dequantization
static inline float get_dequant_value(void *data, TfLiteType tensor_type,
                                      int idx, float zero_point, float scale) {
  switch (tensor_type) {
  case kTfLiteUInt8:
    return (static_cast<uint8_t *>(data)[idx] - zero_point) * scale;
  case kTfLiteFloat32:
    return static_cast<float *>(data)[idx];
  default:
    break;
  }
  return 0.0f;
}

// Guesses the width, height, and channels of a tensor if it were an image.
// Returns false on failure.
static bool tensor_image_dims(const TfLiteTensor *tensor, int *w, int *h, int *c) {
  int n = TfLiteTensorNumDims(tensor);
  int cursor = 0;

  for (int i = 0; i < n; i++) {
    int dim = TfLiteTensorDim(tensor, i);
    if (dim == 0)
      return false;
    if (dim == 1)
      continue;

    switch (cursor++) {
    case 0:
      if (w)
        *w = dim;
      break;
    case 1:
      if (h)
        *h = dim;
      break;
    case 2:
      if (c)
        *c = dim;
      break;
    default:
      return false;
      break;
    }
  }

  // Ensure that we at least have the width and height.
  if (cursor < 2)
    return false;
  // If we don't have the number of channels, then assume there's only one.
  if (cursor == 2 && c)
    *c = 1;
  // Ensure we have no more than 4 image channels.
  if (*c > 4)
    return false;
  // The tensor dimension appears coherent.
  return true;
}

static inline float calculateIoU(const BOX_RECT &box1, const BOX_RECT &box2) {
  // Calculate intersection coordinates
  const int x1 = std::max(box1.left, box2.left);
  const int y1 = std::max(box1.top, box2.top);
  const int x2 = std::min(box1.right, box2.right);
  const int y2 = std::min(box1.bottom, box2.bottom);

  // No intersection case
  if (x2 <= x1 || y2 <= y1)
    return 0.0f;

  // Calculate areas using pre-computed values when possible
  const int intersectionArea = (x2 - x1) * (y2 - y1);
  const int area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
  const int area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

  return static_cast<float>(intersectionArea) /
         (area1 + area2 - intersectionArea);
}

static std::vector<detect_result_t>
optimizedNMS(std::vector<detect_result_t> &candidates, float nmsThreshold) {
  if (candidates.empty())
    return {};

  // Sort by confidence (descending) - single pass
  std::sort(candidates.begin(), candidates.end(),
            [](const detect_result_t &a, const detect_result_t &b) {
              return a.obj_conf > b.obj_conf;
            });

  std::vector<detect_result_t> results;
  results.reserve(candidates.size() / 4); // Reasonable initial capacity

  // Use bitset for faster suppression tracking
  std::vector<bool> suppressed(candidates.size(), false);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;

    // Keep this detection
    results.push_back(candidates[i]);
    const auto &currentBox = candidates[i];

    // Suppress overlapping boxes of the SAME class only
    // Start from i+1 since array is sorted by confidence
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] || candidates[j].id != currentBox.id)
        continue;

      if (calculateIoU(currentBox.box, candidates[j].box) > nmsThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return results;
}

RubikDetector* rubik_create(const char* model_path, char** error_msg) {
  // Load the model
  TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
  if (!model) {
    if (error_msg) *error_msg = strdup("Failed to load model file");
    return nullptr;
  }

  DEBUG_PRINT("INFO: Loaded model file '%s'\n", model_path);

  // Create external delegate options
  TfLiteExternalDelegateOptions delegateOptsValue =
      TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");

  TfLiteExternalDelegateOptions *delegateOpts = &delegateOptsValue;

  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "backend_type",
                                          "htp") != kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Failed to set backend type to htp");
    TfLiteModelDelete(model);
    return nullptr;
  }

  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "htp_use_conv_hmx",
                                          "1") != kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Failed to enable convolutions");
    TfLiteModelDelete(model);
    return nullptr;
  }

  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "htp_performance_mode",
                                          "2") != kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Failed to set htp performance mode");
    TfLiteModelDelete(model);
    return nullptr;
  }

  // Create the delegate
  TfLiteDelegate *delegate = TfLiteExternalDelegateCreate(delegateOpts);

  if (!delegate) {
    if (error_msg) *error_msg = strdup("Failed to create external delegate");
    TfLiteModelDelete(model);
    return nullptr;
  } else {
    DEBUG_PRINT("INFO: Created external delegate\n");
  }

  DEBUG_PRINT("INFO: Loaded external delegate\n");

  // Create interpreter options
  TfLiteInterpreterOptions *interpreterOpts = TfLiteInterpreterOptionsCreate();
  if (!interpreterOpts) {
    if (error_msg) *error_msg = strdup("Failed to create interpreter options");
    TfLiteExternalDelegateDelete(delegate);
    TfLiteModelDelete(model);
    return nullptr;
  }

  TfLiteInterpreterOptionsAddDelegate(interpreterOpts, delegate);

  // Create the interpreter
  TfLiteInterpreter *interpreter =
      TfLiteInterpreterCreate(model, interpreterOpts);
  TfLiteInterpreterOptionsDelete(interpreterOpts);

  if (!interpreter) {
    if (error_msg) *error_msg = strdup("Failed to create interpreter");
    TfLiteExternalDelegateDelete(delegate);
    TfLiteModelDelete(model);
    return nullptr;
  }

  // Modify graph with delegate
  if (TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate) !=
      kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Failed to modify graph with delegate");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    TfLiteModelDelete(model);
    return nullptr;
  } else {
    DEBUG_PRINT("INFO: Modified graph with external delegate\n");
  }

  // Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Failed to allocate tensors");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    TfLiteModelDelete(model);
    return nullptr;
  }

  // Create RubikDetector object
  RubikDetector *detector = new RubikDetector;
  detector->interpreter = interpreter;
  detector->delegate = delegate;
  detector->model = model;

  DEBUG_PRINT("INFO: TensorFlow Lite initialization completed successfully\n");

  return detector;
}

void rubik_destroy(RubikDetector* detector) {
  if (!detector) {
    return;
  }

  if (detector->interpreter)
    TfLiteInterpreterDelete(detector->interpreter);
  if (detector->delegate)
    TfLiteExternalDelegateDelete(detector->delegate);
  if (detector->model)
    TfLiteModelDelete(detector->model);

  delete detector;

  DEBUG_PRINT("INFO: Object Detection instance destroyed successfully\n");
}

std::vector<detect_result_t> rubik_detect(RubikDetector* detector, cv::Mat* input_img, 
                                           double boxThresh, double nmsThreshold, 
                                           char** error_msg) {
  if (!detector) {
    if (error_msg) *error_msg = strdup("Invalid RubikDetector pointer");
    return {};
  }

  if (!detector->interpreter) {
    if (error_msg) *error_msg = strdup("Interpreter not initialized");
    return {};
  }

  TfLiteInterpreter *interpreter = detector->interpreter;

  TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  int in_w, in_h, in_c;
  if (!tensor_image_dims(input, &in_w, &in_h, &in_c)) {
    if (error_msg) *error_msg = strdup("Invalid input tensor shape");
    return {};
  }

  if (!input_img || input_img->empty() || input_img->cols != in_w ||
      input_img->rows != in_h) {
    if (error_msg) *error_msg = strdup("Invalid input image or mismatched dimensions");
    return {};
  }

  cv::Mat rgb;
  if (input_img->channels() == 3) {
    cv::cvtColor(*input_img, rgb, cv::COLOR_BGR2RGB);
  } else {
    if (error_msg) *error_msg = strdup("Input image must be RGB");
    return {};
  }

  std::memcpy(TfLiteTensorData(input), rgb.data, TfLiteTensorByteSize(input));

// Start timer for benchmark
#ifndef NDEBUG
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    if (error_msg) *error_msg = strdup("Interpreter invocation failed");
    return {};
  }

#ifndef NDEBUG
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
  DEBUG_PRINT("INFO: Model execution time: %.2f ms\n", elapsed_time);
#endif

  const TfLiteTensor *boxesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  const TfLiteTensor *scoresTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 1);
  const TfLiteTensor *classesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 2);

  const TfLiteQuantizationParams boxesParams =
      TfLiteTensorQuantizationParams(boxesTensor);
  const TfLiteQuantizationParams scoresParams =
      TfLiteTensorQuantizationParams(scoresTensor);

  const int numBoxes = TfLiteTensorDim(boxesTensor, 1);
  DEBUG_PRINT("INFO: Detected %d boxes\n", numBoxes);

  if (TfLiteTensorType(boxesTensor) != kTfLiteUInt8) {
    if (error_msg) *error_msg = strdup("Expected uint8 tensor type for boxes");
    return {};
  }

  if (TfLiteTensorType(scoresTensor) != kTfLiteUInt8) {
    if (error_msg) *error_msg = strdup("Expected uint8 tensor type for scores");
    return {};
  }

  if (TfLiteTensorType(classesTensor) != kTfLiteUInt8) {
    if (error_msg) *error_msg = strdup("Expected uint8 tensor type for classes");
    return {};
  }

  uint8_t *boxesData = static_cast<uint8_t *>(TfLiteTensorData(boxesTensor));
  uint8_t *scoresData = static_cast<uint8_t *>(TfLiteTensorData(scoresTensor));
  uint8_t *classesData =
      static_cast<uint8_t *>(TfLiteTensorData(classesTensor));

  std::vector<detect_result_t> candidateResults;

  for (int i = 0; i < numBoxes; ++i) {
    float score =
        get_dequant_value(scoresData, kTfLiteUInt8, i, scoresParams.zero_point,
                          scoresParams.scale);
    if (score < boxThresh)
      continue;

    int classId = classesData[i];

    uint8_t raw_x_1_u8 = boxesData[i * 4 + 0];
    uint8_t raw_y_1_u8 = boxesData[i * 4 + 1];
    uint8_t raw_x_2_u8 = boxesData[i * 4 + 2];
    uint8_t raw_y_2_u8 = boxesData[i * 4 + 3];

    float x1 = get_dequant_value(&raw_x_1_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float y1 = get_dequant_value(&raw_y_1_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float x2 = get_dequant_value(&raw_x_2_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float y2 = get_dequant_value(&raw_y_2_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);

    float clamped_x1 =
        std::max(0.0f, std::min(x1, static_cast<float>(input_img->cols)));
    float clamped_y1 =
        std::max(0.0f, std::min(y1, static_cast<float>(input_img->rows)));
    float clamped_x2 =
        std::max(0.0f, std::min(x2, static_cast<float>(input_img->cols)));
    float clamped_y2 =
        std::max(0.0f, std::min(y2, static_cast<float>(input_img->rows)));

    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

    detect_result_t det;
    det.box.left = static_cast<int>(std::round(clamped_x1));
    det.box.top = static_cast<int>(std::round(clamped_y1));
    det.box.right = static_cast<int>(std::round(clamped_x2));
    det.box.bottom = static_cast<int>(std::round(clamped_y2));
    det.obj_conf = score;
    det.id = classId;

    candidateResults.push_back(det);
  }

  // NMS
  std::vector<detect_result_t> results =
      optimizedNMS(candidateResults, static_cast<float>(nmsThreshold));

  return results;
}

bool rubik_is_quantized(RubikDetector* detector, char** error_msg) {
  if (!detector) {
    if (error_msg) *error_msg = strdup("Invalid RubikDetector pointer");
    return false;
  }

  if (!detector->interpreter) {
    if (error_msg) *error_msg = strdup("Interpreter not initialized");
    return false;
  }

  TfLiteInterpreter *interpreter = detector->interpreter;

  // Check if the input tensor is quantized
  TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  if (!input) {
    if (error_msg) *error_msg = strdup("Failed to get input tensor");
    return false;
  }

  // Check if the tensor type is kTfLiteUInt8
  TfLiteType tensorType = TfLiteTensorType(input);

  if (tensorType == kTfLiteUInt8) {
    DEBUG_PRINT("INFO: Input tensor is quantized\n");
    return true;
  } else {
    DEBUG_PRINT("INFO: Input tensor is not quantized\n");
    return false;
  }
}
