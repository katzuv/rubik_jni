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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rubik_core.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;

class RubikDetectorWrapper {
private:
    RubikDetector* detector_;
    
public:
    RubikDetectorWrapper(const std::string& model_path) {
        char* error_msg = nullptr;
        detector_ = rubik_create(model_path.c_str(), &error_msg);
        if (!detector_) {
            std::string err = error_msg ? error_msg : "Unknown error";
            if (error_msg) free(error_msg);
            throw std::runtime_error("Failed to create detector: " + err);
        }
    }
    
    ~RubikDetectorWrapper() {
        if (detector_) {
            rubik_destroy(detector_);
        }
    }
    
    // Prevent copying
    RubikDetectorWrapper(const RubikDetectorWrapper&) = delete;
    RubikDetectorWrapper& operator=(const RubikDetectorWrapper&) = delete;
    
    std::vector<detect_result_t> detect(py::array_t<uint8_t> image, double box_thresh, double nms_thresh) {
        auto buf = image.request();
        
        if (buf.ndim != 3) {
            throw std::runtime_error("Image must be 3-dimensional (height, width, channels)");
        }
        
        int height = buf.shape[0];
        int width = buf.shape[1];
        int channels = buf.shape[2];
        
        if (channels != 3) {
            throw std::runtime_error("Image must have 3 channels (BGR)");
        }
        
        // Create cv::Mat from numpy array
        cv::Mat img(height, width, CV_8UC3, (unsigned char*)buf.ptr);
        
        char* error_msg = nullptr;
        auto results = rubik_detect(detector_, &img, box_thresh, nms_thresh, &error_msg);
        
        if (error_msg) {
            std::string err = error_msg;
            free(error_msg);
            throw std::runtime_error("Detection failed: " + err);
        }
        
        return results;
    }
    
    bool is_quantized() {
        char* error_msg = nullptr;
        bool result = rubik_is_quantized(detector_, &error_msg);
        
        if (error_msg) {
            std::string err = error_msg;
            free(error_msg);
            throw std::runtime_error("Failed to check quantization: " + err);
        }
        
        return result;
    }
};

PYBIND11_MODULE(rubik_py, m) {
    m.doc() = "Python bindings for Rubik object detection";
    
    py::class_<BOX_RECT>(m, "BoxRect")
        .def(py::init<>())
        .def_readwrite("left", &BOX_RECT::left)
        .def_readwrite("right", &BOX_RECT::right)
        .def_readwrite("top", &BOX_RECT::top)
        .def_readwrite("bottom", &BOX_RECT::bottom)
        .def("__repr__", [](const BOX_RECT &r) {
            return "BoxRect(left=" + std::to_string(r.left) + 
                   ", top=" + std::to_string(r.top) + 
                   ", right=" + std::to_string(r.right) + 
                   ", bottom=" + std::to_string(r.bottom) + ")";
        });
    
    py::class_<detect_result_t>(m, "DetectResult")
        .def(py::init<>())
        .def_readwrite("id", &detect_result_t::id)
        .def_readwrite("box", &detect_result_t::box)
        .def_readwrite("obj_conf", &detect_result_t::obj_conf)
        .def("__repr__", [](const detect_result_t &r) {
            return "DetectResult(id=" + std::to_string(r.id) + 
                   ", confidence=" + std::to_string(r.obj_conf) + 
                   ", box=BoxRect(left=" + std::to_string(r.box.left) + 
                   ", top=" + std::to_string(r.box.top) + 
                   ", right=" + std::to_string(r.box.right) + 
                   ", bottom=" + std::to_string(r.box.bottom) + "))";
        });
    
    py::class_<RubikDetectorWrapper>(m, "RubikDetector")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("detect", &RubikDetectorWrapper::detect, 
             py::arg("image"), 
             py::arg("box_thresh") = 0.5, 
             py::arg("nms_thresh") = 0.45,
             "Detect objects in an image. Image should be a numpy array with shape (height, width, 3)")
        .def("is_quantized", &RubikDetectorWrapper::is_quantized,
             "Check if the model is quantized");
}
