"""
Face Detection Comparison: OpenCV Haar Cascade vs. Google MediaPipe BlazeFace
=============================================================================
This program demonstrates and compares two popular methods for face detection:
1. Classical Computer Vision: OpenCV Haar Feature-based Cascade Classifier.
2. Modern Deep Learning: Google MediaPipe BlazeFace Short-Range Detector.

Key Differences at a Glance:
- Haar Cascade:
    - Uses hand-crafted edge and line features (the Viola-Jones algorithm).
    - Extremely lightweight and fast on CPU.
    - Sensitive to face angles/rotations, lighting variations, and partial occlusions.
    - Requires single-channel grayscale images.
- MediaPipe BlazeFace:
    - Uses a lightweight Convolutional Neural Network (CNN) optimized for mobile/edge.
    - Highly accurate, robust to facial angles, expressions, and occlusions.
    - Requires 3-channel RGB images.
"""

import os
import sys
import time
import urllib.request
import ctypes
import cv2
import mediapipe as mp

# =============================================================================
# 0. WINDOWS COMPATIBILITY PATCH FOR MEDIAPIPE
# =============================================================================
# On some Windows installations, the MediaPipe native C++ library (libmediapipe.dll)
# cannot locate the standard C runtime memory function 'free'.
# This patch intercepts the library loading process and binds the Windows C runtime
# msvcrt.free to prevent MediaPipe from crashing with an AttributeError.
try:
    from mediapipe.tasks.python.core import mediapipe_c_bindings
    _orig_load_raw = mediapipe_c_bindings.load_raw_library

    def _safe_load_raw_library(signatures=()):
        try:
            return _orig_load_raw(signatures)
        except AttributeError as e:
            if "free" in str(e) and os.name == "nt":
                # Bind standard C runtime free function from msvcrt.dll
                mediapipe_c_bindings._shared_lib.free = ctypes.cdll.msvcrt.free
                mediapipe_c_bindings._shared_lib.free.argtypes = [ctypes.c_void_p]
                mediapipe_c_bindings._shared_lib.free.restype = None
                return mediapipe_c_bindings._shared_lib
            raise

    mediapipe_c_bindings.load_raw_library = _safe_load_raw_library
except Exception:
    pass


# =============================================================================
# 1. PATH CONFIGURATION & MODEL INITIALIZATION
# =============================================================================
# Using the absolute directory of this script ensures all paths work reliably
# regardless of what current working directory the terminal is running from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Haar Cascade Model Setup ---
# The Haar Cascade model is stored as an XML file containing pre-trained decision trees.
HAAR_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

if not os.path.exists(HAAR_PATH):
    # Check if OpenCV already has the Haar XML bundled in its own installation
    cv2_data_path = os.path.join(
        getattr(cv2.data, "haarcascades", ""),
        "haarcascade_frontalface_default.xml"
    )
    if os.path.exists(cv2_data_path):
        HAAR_PATH = cv2_data_path
    else:
        print("[Setup] Downloading Haar cascade XML model (one-time setup)...")
        urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)

# Load the Haar Cascade classifier into OpenCV
haar_detector = cv2.CascadeClassifier(HAAR_PATH)
if haar_detector.empty():
    raise RuntimeError(f"Failed to load Haar cascade classifier from: {HAAR_PATH}")


# --- MediaPipe BlazeFace Model Setup ---
# MediaPipe uses a lightweight TensorFlow Lite (.tflite) deep learning model file.
MODEL_PATH = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

if not os.path.exists(MODEL_PATH):
    print("[Setup] Downloading MediaPipe BlazeFace model (one-time setup)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Configure MediaPipe Face Detector options
# - model_asset_path: path to the downloaded .tflite weights
# - min_detection_confidence: detection threshold (0.5 = 50% confidence minimum)
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
media_detector = mp.tasks.vision.FaceDetector.create_from_options(options)


# =============================================================================
# 2. DETECTION FUNCTIONS
# =============================================================================
def detect_haar(image):
    """
    Detect faces in an image using OpenCV's Haar Cascade classifier.

    How it works for beginners:
    1. Grayscale Conversion:
       Color information (red, green, blue) is not needed by Haar features.
       Converting to a single grayscale channel significantly speeds up computation.
    2. Multi-Scale Sliding Window (detectMultiScale):
       The detector scans across the image using windows of varying sizes to find
       faces that are close (large) or far away (small).
    3. Feature Matching (Viola-Jones):
       Compares pixel brightness differences across rectangular regions to detect
       facial patterns (e.g., eyes are generally darker than the forehead/cheeks).
    4. Bounding Box Annotation:
       Draws green rectangles around all detected face locations.

    Parameters:
        image (numpy.ndarray): The input BGR image from OpenCV.

    Returns:
        output (numpy.ndarray): Copy of the image annotated with green bounding boxes.
        count (int): Number of detected faces.
        elapsed (float): Detection time in seconds.
    """
    # Step 1: Convert BGR (Blue-Green-Red) image to single-channel Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Time the detection process using high-precision performance counter
    start = time.perf_counter()
    
    # detectMultiScale parameters explained:
    # - scaleFactor=1.1: At each step, the image size is scaled down by 10% to find different face sizes.
    # - minNeighbors=5: How many candidate rectangles must overlap to confirm a face (higher = fewer false positives).
    # - minSize=(30, 30): Minimum possible bounding box size in pixels (filters out tiny visual artifacts).
    faces = haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    elapsed = time.perf_counter() - start

    # Step 3: Copy the original image so our drawing does not modify the original data
    output = image.copy()

    # Step 4: Draw bounding boxes for each face found
    # Each item in 'faces' is a tuple: (x, y, width, height)
    # (x, y) is the top-left corner coordinate; (x + w, y + h) is the bottom-right corner.
    for (x, y, w, h) in faces:
        # cv2.rectangle syntax: (image, top_left_point, bottom_right_point, color_bgr, thickness)
        # Note: (0, 255, 0) represents bright GREEN in OpenCV's BGR color format.
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Step 5: Overlay text showing the number of detected faces and processing time
    label = f"Haar: {len(faces)} face(s) ({elapsed * 1000:.1f} ms)"
    cv2.putText(
        output,
        label,
        (10, 30),                  # Text position (x, y) near top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        0.7,                       # Font scale (size)
        (0, 255, 0),               # Text color in BGR (Green)
        2                          # Stroke thickness
    )

    return output, len(faces), elapsed


def detect_mediapipe(image):
    """
    Detect faces in an image using Google MediaPipe BlazeFace.

    How it works for beginners:
    1. Color Space Conversion (BGR to RGB):
       OpenCV stores color as Blue-Green-Red (BGR), but MediaPipe neural networks
       were trained on standard Red-Green-Blue (RGB) images.
    2. MediaPipe Image Wrapper:
       Converts the NumPy array into an optimized `mp.Image` object for the pipeline.
    3. Neural Network Inference:
       Runs the BlazeFace deep learning model, which looks at complex visual patterns
       and produces high-accuracy bounding boxes even if faces are angled or partially hidden.
    4. Bounding Box Extraction & Boundary Clamping:
       Extracts origin (x, y) and dimensions (w, h), clamping them to image bounds
       so coordinates never exceed the image dimensions.
    5. Bounding Box Annotation:
       Draws blue rectangles around detected faces.

    Parameters:
        image (numpy.ndarray): The input BGR image from OpenCV.

    Returns:
        output (numpy.ndarray): Copy of the image annotated with blue bounding boxes.
        count (int): Number of detected faces.
        elapsed (float): Detection time in seconds.
    """
    # Step 1: Convert BGR to RGB format (required by MediaPipe models)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Wrap the RGB array in MediaPipe's Image container
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Step 3: Run deep learning detection and measure time
    start = time.perf_counter()
    results = media_detector.detect(mp_image)
    elapsed = time.perf_counter() - start

    output = image.copy()
    faces = results.detections if results.detections else []

    # Get image dimensions (height, width) to prevent bounding boxes from going out of bounds
    height, width = image.shape[:2]

    # Step 4: Draw bounding boxes for each detection
    for face in faces:
        box = face.bounding_box

        # MediaPipe provides coordinates in pixel values:
        # origin_x: starting x coordinate (left)
        # origin_y: starting y coordinate (top)
        # width, height: dimensions of the bounding box
        x = max(0, int(box.origin_x))
        y = max(0, int(box.origin_y))
        w = int(box.width)
        h = int(box.height)

        # Calculate bottom-right corner and clamp to image limits
        x2 = min(width - 1, x + w)
        y2 = min(height - 1, y + h)

        # Note: (255, 0, 0) represents bright BLUE in OpenCV's BGR color format.
        cv2.rectangle(
            output,
            (x, y),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Step 5: Overlay text with face count and inference time
    label = f"MediaPipe: {len(faces)} face(s) ({elapsed * 1000:.1f} ms)"
    cv2.putText(
        output,
        label,
        (10, 30),                  # Text position (x, y) near top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        0.7,                       # Font scale (size)
        (255, 0, 0),               # Text color in BGR (Blue)
        2                          # Stroke thickness
    )

    return output, len(faces), elapsed


# =============================================================================
# 3. STATIC IMAGE TESTING
# =============================================================================
def test_images():
    """
    Batch tests face detection on all image files inside the 'sample_images' directory.
    - Runs both Haar Cascade and MediaPipe detectors on each image.
    - Prints quantitative benchmark comparison (face counts, processing time in ms).
    - Saves the annotated comparison images into the 'results' folder.
    - Displays interactive preview windows side-by-side for visual inspection.
    """
    # Check if the sample folder exists; if not, create it and guide the user
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
        print(f"[Notice] Created folder '{SAMPLE_IMAGES_DIR}'.")
        print("Please place some test photos (.jpg, .png) there and run this option again.")
        return

    # Find all supported image formats in the folder
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in os.listdir(SAMPLE_IMAGES_DIR)
        if f.lower().endswith(extensions)
    ]

    if not image_files:
        print(f"[Notice] No images found in '{SAMPLE_IMAGES_DIR}'.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp.")
        print("Please add at least one image to test.")
        return

    # Ensure output results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nFound {len(image_files)} image(s) to analyze.")
    print("=" * 60)

    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(SAMPLE_IMAGES_DIR, filename)
        
        # cv2.imread loads the image from disk into a NumPy array
        image = cv2.imread(image_path)

        if image is None:
            print(f"[Warning] Failed to read image '{filename}'. Skipping.")
            continue

        # Run both detectors on the identical image
        haar_img, haar_count, haar_time = detect_haar(image)
        mp_img, mp_count, mp_time = detect_mediapipe(image)

        # Print formatted performance comparison
        print(f"[{index}/{len(image_files)}] File: {filename}")
        print(f"  * Haar Cascade   : {haar_count} face(s) found in {haar_time * 1000:6.2f} ms")
        print(f"  * MediaPipe      : {mp_count} face(s) found in {mp_time * 1000:6.2f} ms")

        # Save annotated result images into the 'results' folder
        cv2.imwrite(os.path.join(RESULTS_DIR, f"haar_{filename}"), haar_img)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"mediapipe_{filename}"), mp_img)

        # Show the comparison windows to the user
        cv2.imshow("Haar Cascade (Classical CV)", haar_img)
        cv2.imshow("MediaPipe (Deep Learning)", mp_img)

        print("  --> Press ANY KEY to inspect next image (or 'q' / ESC to return to menu)...")
        # cv2.waitKey(0) waits indefinitely until any keyboard key is pressed
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # 27 is the ASCII code for ESC
            print("[Info] Inspection stopped by user.")
            break

    # Clean up all OpenCV display windows
    cv2.destroyAllWindows()
    print(f"\nAll processed results are saved in: {RESULTS_DIR}")


# =============================================================================
# 4. REAL-TIME WEBCAM DETECTION
# =============================================================================
def test_webcam():
    """
    Captures live video frames from the webcam and runs both detectors in real time.
    Displays side-by-side comparison windows.
    Press 'q' or 'ESC' to close the camera stream and return to the menu.
    """
    # cv2.VideoCapture(0) opens the default computer camera (index 0)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[Error] Could not access the webcam.")
        print("Please verify that your camera is connected and not in use by another app.")
        return

    print("\nLive webcam detection started.")
    print("Controls: Press 'q' or 'ESC' in any video window to stop.")

    try:
        while True:
            # camera.read() returns:
            # - success: True if a frame was grabbed and decoded correctly
            # - frame: The current video frame as a NumPy array (BGR)
            success, frame = camera.read()

            if not success:
                print("[Warning] Could not read frame from webcam.")
                break

            # Run both detectors on the current frame
            haar_frame, _, _ = detect_haar(frame)
            mp_frame, _, _ = detect_mediapipe(frame)

            # Display real-time output in separate windows
            cv2.imshow("Webcam - Haar Cascade (OpenCV)", haar_frame)
            cv2.imshow("Webcam - MediaPipe (Deep Learning)", mp_frame)

            # cv2.waitKey(1) waits 1 millisecond for key press
            # & 0xFF extracts the last 8 bits for cross-platform keyboard compatibility
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or ESC
                print("Stopping webcam feed...")
                break
    finally:
        # Crucial cleanup: release hardware camera handle and destroy GUI windows
        # Using try...finally ensures these are executed even if an unexpected error occurs
        camera.release()
        cv2.destroyAllWindows()


# =============================================================================
# 5. MAIN PROGRAM LOOP
# =============================================================================
def main():
    """
    Main interactive menu loop.
    Allows user to select between image testing, live webcam, or exiting.
    """
    try:
        while True:
            print("\n" + "=" * 50)
            print("         FACE DETECTION COMPARISON BENCHMARK      ")
            print("=" * 50)
            print("1. Test static images (from 'sample_images' folder)")
            print("2. Test real-time webcam feed")
            print("3. Exit")
            print("-" * 50)

            choice = input("Enter choice (1, 2, or 3): ").strip()

            if choice == "1":
                test_images()
            elif choice == "2":
                test_webcam()
            elif choice in ("3", "q", "exit", "quit"):
                print("Exiting program. Goodbye!")
                break
            else:
                print("[Error] Invalid choice. Please enter 1, 2, or 3.")
    finally:
        # Gracefully release MediaPipe native resources
        try:
            media_detector.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
