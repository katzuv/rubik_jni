# Shared Libraries

Last build of tensorflow libraries was on version 2.19.0
To build the necessary tensorflow libraries, the following commands can be used

```
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/external:external_delegate
```

# Building On a Rubik Pi

```
sudo apt-get update
sudo apt-get install -y build-essential cmake openjdk-17-jdk default-jdk
git clone https://github.com/PhotonVision/rubik_jni
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=cmake_build -DOPENCV_ARCH=linuxarm64 ; cmake --build cmake_build --target install -- -j 4
# Sudo is needed because of some weird permission issues with gradle and native libraries
sudo ./gradlew build -x spotlessCheck
```

# Python Bindings

This repository now includes Python bindings that provide access to the same object detection functionality available through the JNI interface.

## Building Python Bindings

The Python bindings are built automatically when you build the project with CMake:

```bash
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Release -DOPENCV_ARCH=linuxarm64
cmake --build cmake_build -- -j 4
```

To disable Python bindings, add `-DBUILD_PYTHON_BINDINGS=OFF` to the cmake command.

The Python module (`rubik_py.so`) will be built in the `cmake_build` directory.

## Using Python Bindings

```python
import rubik_py
import cv2

# Load model
detector = rubik_py.RubikDetector("path/to/model.tflite")

# Check if model is quantized
print(f"Model is quantized: {detector.is_quantized()}")

# Load image (BGR format, as returned by cv2.imread)
image = cv2.imread("path/to/image.jpg")

# Run detection
detections = detector.detect(image, box_thresh=0.5, nms_thresh=0.45)

# Process results
for det in detections:
    print(f"Class: {det.id}, Confidence: {det.obj_conf:.2f}")
    print(f"Box: ({det.box.left}, {det.box.top}) to ({det.box.right}, {det.box.bottom})")
```

See `example_python.py` for a complete working example.

## MemLeak test

Build and run repeated iterations of creating and destroying a detector to find memory leaks. It's necessary to watch memory by hand, as the test only runs repeated create and destroy cycles, it doesn't monitor memory.

```
./gradlew build -PmemLeakTestIterations=1000
```

## Benchmark

Run the detect function repeatedly to benchmark performance.

```
./gradlew build -PbenchmarkIterations=1000
```
