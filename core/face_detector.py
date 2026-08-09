"""
Face detection using OpenCV's bundled Haar Cascade classifier.
"""
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FaceDetector:
    def __init__(self):
        cascade_file = os.path.join(cv2.data.haarcascades, config.FACE_CASCADE_PATH)
        self.cascade = cv2.CascadeClassifier(cascade_file)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_file}")

    def detect_faces(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.DETECTOR_SCALE_FACTOR,
            minNeighbors=config.DETECTOR_MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE,
        )
        return faces

    def extract_face(self, frame_bgr, box):
        x, y, w, h = box
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, config.FACE_SIZE)
        face = cv2.equalizeHist(face)
        return face
