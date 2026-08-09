"""
Trains the LBPH face recognizer on every enrolled person and saves the model.
Run after every enroll.py call.
"""
import os
import sys
import pickle
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from database.db_manager import DBManager


def main():
    db = DBManager()
    persons = db.get_all_persons()

    if not persons:
        print("No enrolled people found. Run enroll.py first.")
        return

    faces, labels = [], []
    label_to_person = {}

    for person_id, label_id, name, person_type, department in persons:
        person_dir = os.path.join(config.KNOWN_FACES_DIR, person_id)
        if not os.path.isdir(person_dir):
            print(f"  WARNING: no image folder for {person_id}, skipping")
            continue

        label_to_person[label_id] = person_id
        img_count = 0
        for fname in os.listdir(person_dir):
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, config.FACE_SIZE)
            faces.append(img)
            labels.append(label_id)
            img_count += 1

        print(f"  {name} ({person_id}, {person_type}): {img_count} samples")

    if not faces:
        print("No usable face images found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    recognizer.save(config.TRAINED_MODEL_PATH)
    with open(config.LABELS_PATH, "wb") as f:
        pickle.dump(label_to_person, f)

    print(f"\nTraining complete: {len(faces)} total samples, {len(label_to_person)} people.")
    print(f"Model saved to {config.TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
