"""
Central configuration for the Face Recognition Attendance System (OpenCV/LBPH version).
No TensorFlow dependency - works on ANY Python version, including 3.14.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")   # one subfolder per person
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# Database
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")

# Face detection (Haar Cascade, bundled with OpenCV)
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"  # resolved via cv2.data.haarcascades
FACE_SIZE = (200, 200)          # size all detected faces are resized to before train/predict
MIN_FACE_SIZE = (60, 60)        # ignore detections smaller than this (reduces false positives)
DETECTOR_SCALE_FACTOR = 1.1
DETECTOR_MIN_NEIGHBORS = 6

# Recognition
# LBPH predict() returns a distance: LOWER = more confident match.
# Tune this after testing on your own dataset (typical usable range: 40-70).
# NOTE: LBPH is sensitive to lighting/angle/location changes between enrollment and
# recognition. If cross-location accuracy is critical, use the TensorFlow/DeepFace
# version instead - it's built specifically for that.
CONFIDENCE_THRESHOLD = 50.0

# Enrollment
SAMPLES_PER_PERSON = 30   # more samples = better accuracy; vary angle/lighting between shots

# Attendance
ATTENDANCE_COOLDOWN_MINUTES = 0  # 0 = only one "present" mark per person per day
