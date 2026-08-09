"""
Enrollment: register a new person and capture face samples.
Usage:
    python enroll.py --id STU-001 --name "Priya Shah" --type student --department "CS-A" --webcam
    python enroll.py --id STU-002 --name "Aman Patel" --type student --department "CS-B" --from-folder ./photos/aman
"""
import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from core.face_detector import FaceDetector
from database.db_manager import DBManager


def capture_from_webcam(person_dir, detector, samples_needed):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not access webcam. Use --from-folder instead.")
        return 0

    count = 0
    print("Look at the camera. Capturing samples... press 'q' to stop early.")
    while count < samples_needed:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.detect_faces(frame)
        for box in faces:
            face = detector.extract_face(frame, box)
            count += 1
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
        cv2.putText(frame, f"Samples: {count}/{samples_needed}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Enrollment - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def capture_from_folder(person_dir, detector, source_folder):
    count = 0
    for fname in sorted(os.listdir(source_folder)):
        path = os.path.join(source_folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            print(f"  no face found in {fname}, skipping")
            continue
        face = detector.extract_face(img, faces[0])
        count += 1
        cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face)
    return count


def main():
    parser = argparse.ArgumentParser(description="Enroll a new person into the attendance system.")
    parser.add_argument("--id", required=True, help="Unique person ID, e.g. STU-001 or EMP-014")
    parser.add_argument("--name", required=True, help="Full name")
    parser.add_argument("--type", required=True, choices=["student", "employee"])
    parser.add_argument("--department", default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--from-folder", metavar="PATH")
    args = parser.parse_args()

    db = DBManager()
    if db.person_exists(args.id):
        print(f"ERROR: person_id '{args.id}' already enrolled.")
        return

    detector = FaceDetector()
    person_dir = os.path.join(config.KNOWN_FACES_DIR, args.id)
    os.makedirs(person_dir, exist_ok=True)

    if args.from_folder:
        count = capture_from_folder(person_dir, detector, args.from_folder)
    elif args.webcam:
        count = capture_from_webcam(person_dir, detector, config.SAMPLES_PER_PERSON)
    else:
        print("ERROR: specify either --webcam or --from-folder <path>")
        return

    if count == 0:
        print("No face samples captured - enrollment aborted.")
        os.rmdir(person_dir)
        return

    db.add_person(args.id, args.name, args.type, args.department)
    print(f"Enrolled {args.name} ({args.id}, {args.type}) with {count} face samples.")
    print("Now run: python train_model.py")


if __name__ == "__main__":
    main()
