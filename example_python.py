#!/usr/bin/env python3
"""
Simple example demonstrating how to use the rubik_py Python bindings.

This example shows how to:
1. Load a TensorFlow Lite model
2. Load an image using OpenCV (if available) or PIL
3. Run object detection
4. Display results
"""

import sys
import numpy as np

try:
    import cv2
    HAVE_OPENCV = True
except ImportError:
    HAVE_OPENCV = False
    print("Warning: OpenCV not available, using PIL instead")
    try:
        from PIL import Image
    except ImportError:
        print("Error: Neither OpenCV nor PIL is available. Please install one of them.")
        sys.exit(1)

try:
    import rubik_py
except ImportError:
    print("Error: rubik_py module not found. Please build it first using CMake.")
    print("See README.md for build instructions.")
    sys.exit(1)


def load_image(image_path):
    """Load an image from file and return as numpy array."""
    if HAVE_OPENCV:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    else:
        # Using PIL
        img = Image.open(image_path)
        img = img.convert('RGB')
        # Convert to BGR for consistency with OpenCV
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        return img_array


def draw_detections(image, detections):
    """Draw bounding boxes on the image."""
    if not HAVE_OPENCV:
        print("OpenCV not available, skipping visualization")
        return image
    
    for det in detections:
        box = det.box
        # Draw rectangle
        cv2.rectangle(image, 
                     (box.left, box.top), 
                     (box.right, box.bottom), 
                     (0, 255, 0), 2)
        
        # Put label
        label = f"Class {det.id}: {det.obj_conf:.2f}"
        cv2.putText(image, label, 
                   (box.left, box.top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 2)
    
    return image


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 example_python.py <model_path> <image_path> [box_thresh] [nms_thresh]")
        print("Example: python3 example_python.py model.tflite image.jpg 0.5 0.45")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    box_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    nms_thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
    
    print(f"Loading model from: {model_path}")
    detector = rubik_py.RubikDetector(model_path)
    
    print(f"Model is quantized: {detector.is_quantized()}")
    
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    print(f"Running detection (box_thresh={box_thresh}, nms_thresh={nms_thresh})...")
    detections = detector.detect(image, box_thresh, nms_thresh)
    
    print(f"Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det}")
    
    # Draw and save results
    if HAVE_OPENCV:
        output_image = draw_detections(image.copy(), detections)
        output_path = "detection_results.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
