"""
Enrollment: register a new person and capture face samples.

Usage:
    Interactive (recommended):
        python enroll.py
        (Will prompt for ID, Name, Type, Department, confirm webcam readiness, and capture faces)

    Command Line:
        python enroll.py --id STU-001 --name "Priya Shah" --type student --department "CS-A" --webcam
        python enroll.py --id STU-002 --name "Aman Patel" --type student --department "CS-B" --from-folder ./photos/aman
"""
import argparse
import os
import sys
import time
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from core.face_detector import FaceDetector
from database.db_manager import DBManager


def capture_from_webcam(person_dir, detector, samples_needed):
    """
    Opens the webcam and captures face samples with visual feedback and spacing
    between captures to allow varied facial angles.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\nERROR: Could not access webcam. Please check if your camera is connected")
        print("       and not currently being used by another application.")
        return 0

    count = 0
    last_capture_time = 0.0
    capture_interval = 0.2  # 200ms spacing between samples for natural variation

    print("\nStarting webcam...")
    print(f"Please look at the camera. Target: {samples_needed} face samples.")
    print("Move your head slightly (different angles/expressions) for best accuracy.")
    print("Press 'q' in the camera window to stop early.\n")

    while count < samples_needed:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to grab frame from webcam.")
            break

        current_time = time.time()
        faces = detector.detect_faces(frame)

        face_detected = False
        for box in faces:
            x, y, w, h = box
            # Green rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_detected = True

            if current_time - last_capture_time >= capture_interval:
                face = detector.extract_face(frame, box)
                count += 1
                cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face)
                last_capture_time = current_time
                print(f"  [+] Captured sample {count}/{samples_needed}")
            break

        # UI Overlay: Top banner with counter
        overlay_color = (0, 255, 0) if face_detected else (0, 165, 255)
        status_text = f"Samples: {count}/{samples_needed}"
        if not face_detected:
            status_text += " (Looking for face...)"

        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 50), (25, 25, 25), cv2.FILLED)
        cv2.putText(frame, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, overlay_color, 2)

        # UI Overlay: Bottom instruction banner
        cv2.rectangle(frame, (10, frame.shape[0] - 40), (frame.shape[1] - 10, frame.shape[0] - 10), (25, 25, 25), cv2.FILLED)
        cv2.putText(
            frame,
            "Press 'q' to stop early | Turn head slightly for varied angles",
            (20, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        cv2.imshow("Enrollment - Face Capture (Press 'q' to finish)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def capture_from_folder(person_dir, detector, source_folder):
    """Loads face images from an existing folder."""
    count = 0
    if not os.path.isdir(source_folder):
        print(f"ERROR: folder not found: {source_folder}")
        return 0

    for fname in sorted(os.listdir(source_folder)):
        path = os.path.join(source_folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            print(f"  No face found in {fname}, skipping")
            continue
        face = detector.extract_face(img, faces[0])
        count += 1
        cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face)
    return count


def prompt_interactive_enrollment(db: DBManager):
    """
    Guides the user step-by-step through enrollment questions:
    ID -> Name -> Type -> Dept -> Ready for webcam confirmation.
    """
    print("\n" + "=" * 46)
    print("       Face Recognition - Person Enrollment   ")
    print("=" * 46)

    # 1. Ask ID
    is_update = False
    existing_person = None
    while True:
        person_id = input("\nEnter ID (e.g., STD-001 or EMP-001): ").strip()
        if not person_id:
            print("  [!] ID cannot be empty. Please enter a valid ID.")
            continue

        if db.person_exists(person_id):
            existing_person = db.get_person_by_id(person_id)
            existing_name = existing_person[2] if existing_person else "Unknown"
            print(f"  [i] Person ID '{person_id}' is already enrolled ({existing_name}).")
            re_enroll = input("      Do you want to re-enroll / update this person? (y/n) [n]: ").strip().lower()
            if re_enroll in ("y", "yes"):
                is_update = True
                break
            else:
                continue
        break

    # 2. Ask Name
    default_name = existing_person[2] if (is_update and existing_person) else ""
    prompt_name = f"Enter Name [{default_name}]: " if default_name else "Enter Name: "
    while True:
        name = input(prompt_name).strip()
        if not name and default_name:
            name = default_name
            break
        if not name:
            print("  [!] Name cannot be empty. Please enter a valid name.")
            continue
        break

    # 3. Ask Type
    default_type = existing_person[3] if (is_update and existing_person) else "student"
    while True:
        type_input = input(f"Enter Type (student / employee) [{default_type}]: ").strip().lower()
        if not type_input:
            person_type = default_type
            break
        if type_input in ("student", "s", "stu"):
            person_type = "student"
            break
        elif type_input in ("employee", "e", "emp", "staff"):
            person_type = "employee"
            break
        else:
            print("  [!] Invalid choice. Please enter 'student' or 'employee'.")

    # 4. Ask Department
    default_dept = (existing_person[4] or "") if (is_update and existing_person) else ""
    prompt_dept = f"Enter Department [{default_dept}]: " if default_dept else "Enter Department (optional, press Enter to skip): "
    dept_input = input(prompt_dept).strip()
    if not dept_input and default_dept:
        department = default_dept
    elif not dept_input:
        department = None
    else:
        department = dept_input

    # Summary
    print("\n" + "-" * 46)
    print("Enrollment Summary:")
    print(f"  - ID:         {person_id}")
    print(f"  - Name:       {name}")
    print(f"  - Type:       {person_type}")
    print(f"  - Department: {department or 'N/A'}")
    print("-" * 46)

    # 5. Ask Ready to webcam
    ready = input("\nReady for webcam? (y/n) [y]: ").strip().lower()
    if ready and ready not in ("y", "yes"):
        print("\nEnrollment cancelled by user.")
        return None

    return {
        "id": person_id,
        "name": name,
        "type": person_type,
        "department": department,
        "is_update": is_update,
    }


def main():
    parser = argparse.ArgumentParser(description="Enroll a new person into the attendance system.")
    parser.add_argument("--id", default=None, help="Unique person ID, e.g. STU-001 or EMP-014")
    parser.add_argument("--name", default=None, help="Full name")
    parser.add_argument("--type", choices=["student", "employee"], default=None)
    parser.add_argument("--department", default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--from-folder", metavar="PATH")
    args, _ = parser.parse_known_args()

    db = DBManager()

    # If --id is not provided via CLI, run interactive questionnaire
    if not args.id:
        info = prompt_interactive_enrollment(db)
        if info is None:
            return
        person_id = info["id"]
        name = info["name"]
        person_type = info["type"]
        department = info["department"]
        is_update = info["is_update"]
        use_webcam = True
        from_folder = None
    else:
        # Command-line arguments provided
        person_id = args.id
        is_update = db.person_exists(person_id)
        if is_update and not args.from_folder and not args.webcam:
            print(f"ERROR: person_id '{person_id}' already enrolled.")
            return

        name = args.name or input("Enter Name: ").strip()
        person_type = args.type or "student"
        department = args.department
        use_webcam = args.webcam or not args.from_folder
        from_folder = args.from_folder

        if use_webcam:
            ready = input("\nReady for webcam? (y/n) [y]: ").strip().lower()
            if ready and ready not in ("y", "yes"):
                print("Enrollment cancelled.")
                return

    detector = FaceDetector()
    person_dir = os.path.join(config.KNOWN_FACES_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    # Clean old photos if updating
    if is_update:
        for f in os.listdir(person_dir):
            if f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(person_dir, f))
                except OSError:
                    pass

    # Capture face samples
    if from_folder:
        count = capture_from_folder(person_dir, detector, from_folder)
    else:
        count = capture_from_webcam(person_dir, detector, config.SAMPLES_PER_PERSON)

    if count == 0:
        print("\nNo face samples captured - enrollment aborted.")
        if os.path.exists(person_dir) and not os.listdir(person_dir):
            try:
                os.rmdir(person_dir)
            except OSError:
                pass
        return

    # Update or add to database
    if is_update:
        db.update_person(person_id, name, person_type, department)
        print(f"\n[OK] Updated '{name}' ({person_id}, {person_type}) with {count} face samples.")
    else:
        db.add_person(person_id, name, person_type, department)
        print(f"\n[OK] Enrolled '{name}' ({person_id}, {person_type}) with {count} face samples.")

    print("\nNext step: Run 'python train_model.py' to train the model.")


if __name__ == "__main__":
    main()
