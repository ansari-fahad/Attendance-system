"""
Loads the trained LBPH model and resolves a detected face to a person's identity
(student or employee), pulling name/type/department from the database.
"""
import os
import sys
import pickle
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.db_manager import DBManager


class FaceRecognizer:
    def __init__(self):
        if not os.path.exists(config.TRAINED_MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Enroll people with enroll.py, then run train_model.py."
            )
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read(config.TRAINED_MODEL_PATH)

        with open(config.LABELS_PATH, "rb") as f:
            self.label_to_person = pickle.load(f)

        self.db = DBManager()

    def identify(self, face_gray):
        label_id, distance = self.model.predict(face_gray)

        if distance > config.CONFIDENCE_THRESHOLD:
            return {"recognized": False, "distance": distance}

        person_id = self.label_to_person.get(label_id)
        if person_id is None:
            return {"recognized": False, "distance": distance}

        record = self.db.get_person_by_label(label_id)
        if record is None:
            return {"recognized": False, "distance": distance}

        name, person_type, department = record[1], record[2], record[3]
        return {
            "recognized": True,
            "person_id": person_id,
            "name": name,
            "type": person_type,
            "department": department,
            "distance": distance,
        }
