"""
Main runtime loop: opens webcam, detects faces, identifies people, marks attendance.
Run: python mark_attendance.py
"""
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from core.face_detector import FaceDetector
from core.recognizer import FaceRecognizer
from database.db_manager import DBManager


def draw_label(frame, box, text, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - 25), (x + w, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    db = DBManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not access webcam.")
        return

    marked_this_session = set()
    print("Attendance system running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for box in detector.detect_faces(frame):
            face = detector.extract_face(frame, box)
            result = recognizer.identify(face)

            if not result["recognized"]:
                draw_label(frame, box, "Unknown", (0, 0, 255))
                continue

            person_id = result["person_id"]
            label_text = f"{result['name']} ({result['type']})"

            if person_id not in marked_this_session:
                if db.already_marked_today(person_id):
                    marked_this_session.add(person_id)
                    label_text += " - already marked"
                else:
                    db.mark_attendance(person_id, result["distance"])
                    marked_this_session.add(person_id)
                    label_text += " - MARKED"
                    print(f"Attendance marked: {result['name']} ({result['type']}, {person_id})")

            draw_label(frame, box, label_text, (0, 200, 0))

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
