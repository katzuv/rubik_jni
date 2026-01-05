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

/*
 * NOTE INTELLISENSE WILL NOT WORK UNTIL THE PROJECT IS BUILT AT LEAST ONCE
 */

#include <jni.h>
#include "rubik_core.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// JNI class reference (this can be global since it's shared)
static jclass detectionResultClass = nullptr;
static jclass runtimeExceptionClass = nullptr;

extern "C" {
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Cache exception classes
  jclass localRuntimeClass = env->FindClass("java/lang/RuntimeException");
  if (localRuntimeClass) {
    runtimeExceptionClass = (jclass)env->NewGlobalRef(localRuntimeClass);
    env->DeleteLocalRef(localRuntimeClass);
  }

  // Find the detection result class
  jclass localClass =
      env->FindClass("org/photonvision/rubik/RubikJNI$RubikResult");
  if (!localClass) {
    std::printf(
        "Couldn't find class org/photonvision/rubik/RubikJNI$RubikResult!\n");
    return JNI_ERR;
  }

  // Create global reference
  detectionResultClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);

  if (!detectionResultClass) {
    std::printf("Couldn't create global reference to class!\n");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
    if (detectionResultClass) {
      env->DeleteGlobalRef(detectionResultClass);
      detectionResultClass = nullptr;
    }
  }
}

static jobject MakeJObject(JNIEnv *env, const detect_result_t &result) {
  if (!detectionResultClass) {
    std::printf("ERROR: detectionResultClass is null!\n");
    return nullptr;
  }

  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");
  if (!constructor) {
    std::printf("ERROR: Could not find constructor for RubikResult!\n");
    return nullptr;
  }

  return env->NewObject(detectionResultClass, constructor, result.box.left,
                        result.box.top, result.box.right, result.box.bottom,
                        result.obj_conf, result.id);
}

// Helper function to throw exceptions
void ThrowRuntimeException(JNIEnv *env, const char *message) {
  if (runtimeExceptionClass) {
    env->ThrowNew(runtimeExceptionClass, message);
  }
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    create
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_rubik_RubikJNI_create
  (JNIEnv *env, jobject obj, jstring modelPath)
{
  const char *model_name = env->GetStringUTFChars(modelPath, nullptr);
  if (model_name == nullptr) {
    ThrowRuntimeException(env, "Failed to retrieve model path");
    return 0;
  }

  char* error_msg = nullptr;
  RubikDetector* detector = rubik_create(model_name, &error_msg);
  
  env->ReleaseStringUTFChars(modelPath, model_name);

  if (!detector) {
    if (error_msg) {
      ThrowRuntimeException(env, error_msg);
      free(error_msg);
    } else {
      ThrowRuntimeException(env, "Unknown error creating detector");
    }
    return 0;
  }

  return reinterpret_cast<jlong>(detector);
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_rubik_RubikJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
  RubikDetector *detector = reinterpret_cast<RubikDetector *>(ptr);
  
  if (!detector) {
    ThrowRuntimeException(env, "Invalid RubikDetector pointer");
    return;
  }

  rubik_destroy(detector);
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    detect
 * Signature: (JJDD)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_rubik_RubikJNI_detect
  (JNIEnv *env, jobject obj, jlong ptr, jlong input_cvmat_ptr,
   jdouble boxThresh, jdouble nmsThreshold)
{
  RubikDetector *detector = reinterpret_cast<RubikDetector *>(ptr);
  cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);

  char* error_msg = nullptr;
  std::vector<detect_result_t> results = rubik_detect(detector, input_img, boxThresh, nmsThreshold, &error_msg);

  if (error_msg) {
    ThrowRuntimeException(env, error_msg);
    free(error_msg);
    return nullptr;
  }

  jobjectArray jResults =
      env->NewObjectArray(results.size(), detectionResultClass, nullptr);
  for (size_t i = 0; i < results.size(); ++i) {
    jobject jDet = MakeJObject(env, results[i]);
    env->SetObjectArrayElement(jResults, i, jDet);
  }

  return jResults;
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    isQuantized
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_org_photonvision_rubik_RubikJNI_isQuantized
  (JNIEnv *env, jobject obj, jlong ptr)
{
  RubikDetector *detector = reinterpret_cast<RubikDetector *>(ptr);

  char* error_msg = nullptr;
  bool is_quantized = rubik_is_quantized(detector, &error_msg);

  if (error_msg) {
    ThrowRuntimeException(env, error_msg);
    free(error_msg);
    return JNI_FALSE;
  }

  return is_quantized ? JNI_TRUE : JNI_FALSE;
}
} // extern "C"
