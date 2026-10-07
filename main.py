"""
Main Controller & CLI Dashboard for the Face Recognition Attendance System.

Provides an interactive menu with options to:
    1: Add / Enroll a Person (capture face samples)
    2: Display (view enrolled persons and attendance logs)
    3: Train (train the LBPH face recognizer model)
    4: Mark Attendance (launch real-time face recognition via webcam)
    5: Exit
"""

import os
import sys
import datetime
import cv2

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from database.db_manager import DBManager
import enroll
import train_model
import mark_attendance


def clear_screen():
    """Clear terminal screen for a cleaner CLI interface."""
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    """Prints the system header banner."""
    print("=" * 64)
    print("       FACE RECOGNITION ATTENDANCE MANAGEMENT SYSTEM       ")
    print("=" * 64)


def display_enrolled_persons(db: DBManager):
    """Displays all enrolled persons in a formatted table."""
    persons = db.get_all_persons()
    print("\n" + "-" * 72)
    print(f"{'REGISTERED PERSONS':^72}")
    print("-" * 72)

    if not persons:
        print("  [!] No registered persons found in database.")
        print("      Choose Option 1 from the main menu to enroll a person.")
        print("-" * 72)
        return

    header = f"{'#':<4} | {'Person ID':<12} | {'Name':<20} | {'Type':<10} | {'Dept':<10} | {'Photos':<6}"
    print(header)
    print("-" * 72)

    students_count = 0
    employees_count = 0

    for idx, (p_id, lbl, name, p_type, dept) in enumerate(persons, start=1):
        if p_type == "student":
            students_count += 1
        elif p_type == "employee":
            employees_count += 1

        dept_str = dept if dept else "-"
        # Count captured photos
        p_dir = os.path.join(config.KNOWN_FACES_DIR, p_id)
        if os.path.isdir(p_dir):
            photo_count = len([f for f in os.listdir(p_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            photo_count = 0

        print(f"{idx:<4} | {p_id:<12} | {name:<20} | {p_type:<10} | {dept_str:<10} | {photo_count:<6}")

    print("-" * 72)
    print(f"Total: {len(persons)} person(s) | Students: {students_count} | Employees: {employees_count}")
    print("-" * 72)


def display_attendance_records(db: DBManager, date_str: str = None):
    """Displays attendance records for a given date in a formatted table."""
    target_date = date_str or datetime.date.today().isoformat()
    records = db.get_attendance_by_date(target_date)

    print("\n" + "-" * 88)
    title = f"ATTENDANCE RECORDS ({target_date})"
    print(f"{title:^88}")
    print("-" * 88)

    if not records:
        print(f"  [i] No attendance records found for date: {target_date}")
        print("-" * 88)
        return

    header = f"{'#':<4} | {'Time':<10} | {'Person ID':<12} | {'Name':<20} | {'Type':<10} | {'Dept':<10} | {'Status':<8} | {'Conf':<6}"
    print(header)
    print("-" * 88)

    for idx, (rec_date, rec_time, p_id, name, p_type, dept, status, conf) in enumerate(records, start=1):
        dept_str = dept if dept else "-"
        conf_str = f"{conf:.1f}" if conf is not None else "-"
        print(f"{idx:<4} | {rec_time:<10} | {p_id:<12} | {name:<20} | {p_type:<10} | {dept_str:<10} | {status:<8} | {conf_str:<6}")

    print("-" * 88)
    print(f"Total Present: {len(records)}")
    print("-" * 88)


def handle_add():
    """Option 1: Add a new person and capture face images."""
    print("\n" + "=" * 64)
    print("                  [1] ADD / ENROLL PERSON                  ")
    print("=" * 64)

    # Reset sys.argv to avoid conflicting CLI arguments passed to main.py
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        enroll.main()
    except Exception as e:
        print(f"\n[!] An error occurred during enrollment: {e}")
    finally:
        sys.argv = orig_argv

    # Check if user wants to train right away after enrolling
    print("\n" + "-" * 64)
    train_prompt = input("Would you like to train the model now with updated faces? (y/n) [y]: ").strip().lower()
    if train_prompt in ("", "y", "yes"):
        handle_train(auto_prompt=False)


def handle_display():
    """Option 2: Display menu for enrolled persons and attendance records."""
    db = DBManager()

    while True:
        print("\n" + "=" * 64)
        print("                     [2] DISPLAY RECORDS                   ")
        print("=" * 64)
        print("  1. Display All Enrolled Persons")
        print("  2. Display Today's Attendance")
        print("  3. Display Attendance by Specific Date")
        print("  4. Display Both (Enrolled Persons + Today's Attendance)")
        print("  5. Back to Main Menu")
        print("=" * 64)

        sub_choice = input("Enter choice (1-5): ").strip()

        if sub_choice == "1":
            display_enrolled_persons(db)
        elif sub_choice == "2":
            display_attendance_records(db)
        elif sub_choice == "3":
            date_input = input("\nEnter Date (YYYY-MM-DD) [Default: Today]: ").strip()
            if not date_input:
                date_input = datetime.date.today().isoformat()
            else:
                # Basic validation
                try:
                    datetime.date.fromisoformat(date_input)
                except ValueError:
                    print("  [!] Invalid date format. Please use YYYY-MM-DD (e.g., 2026-10-07).")
                    continue
            display_attendance_records(db, date_input)
        elif sub_choice == "4":
            display_enrolled_persons(db)
            display_attendance_records(db)
        elif sub_choice in ("5", "b", "back", "q", "exit"):
            break
        else:
            print("  [!] Invalid selection. Please enter a number between 1 and 5.")

        input("\nPress Enter to continue...")


def handle_train(auto_prompt: bool = True):
    """Option 3: Train the face recognizer model."""
    print("\n" + "=" * 64)
    print("               [3] TRAIN FACE RECOGNITION MODEL            ")
    print("=" * 64)

    db = DBManager()
    persons = db.get_all_persons()
    if not persons:
        print("\n[!] No enrolled persons found in database.")
        print("    Please use Option 1 to enroll at least one person first.")
        return

    try:
        train_model.main()
    except Exception as e:
        print(f"\n[!] Training failed with error: {e}")


def handle_mark_attendance():
    """Option 4: Mark attendance via live webcam."""
    print("\n" + "=" * 64)
    print("              [4] MARK ATTENDANCE (LIVE WEBCAM)            ")
    print("=" * 64)

    # Check if trained model and labels exist
    if not os.path.exists(config.TRAINED_MODEL_PATH) or not os.path.exists(config.LABELS_PATH):
        print("\n[!] No trained face recognition model found.")
        print(f"    Expected model at: {config.TRAINED_MODEL_PATH}")
        print("    Please add persons (Option 1) and train the model (Option 3) first.")
        train_now = input("\nWould you like to train the model now? (y/n) [y]: ").strip().lower()
        if train_now in ("", "y", "yes"):
            handle_train(auto_prompt=False)
            if not os.path.exists(config.TRAINED_MODEL_PATH):
                return
        else:
            return

    print("\nStarting live attendance tracking...")
    print("Instructions:")
    print("  - Look into the camera.")
    print("  - Once recognized, your attendance will be marked in the database.")
    print("  - Press 'q' in the camera window to stop and return to menu.\n")

    try:
        mark_attendance.main()
    except Exception as e:
        print(f"\n[!] Attendance tracking encountered an error: {e}")
    finally:
        cv2.destroyAllWindows()


def main():
    """Main application loop."""
    while True:
        print_banner()
        print("  [1] Add Person (Enroll & Capture Face Samples)")
        print("  [2] Display (Enrolled Persons & Attendance Records)")
        print("  [3] Train Model (Train LBPH Face Recognizer)")
        print("  [4] Mark Attendance (Live Webcam Detection)")
        print("  [5] Exit")
        print("=" * 64)

        try:
            choice = input("Select an option (1-5): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting Attendance System. Goodbye!")
            sys.exit(0)

        if choice == "1":
            handle_add()
        elif choice == "2":
            handle_display()
        elif choice == "3":
            handle_train()
        elif choice == "4":
            handle_mark_attendance()
        elif choice in ("5", "q", "exit", "quit"):
            print("\nThank you for using the Face Recognition Attendance System. Goodbye!")
            break
        else:
            print("\n  [!] Invalid choice. Please enter 1, 2, 3, 4, or 5.")

        input("\nPress Enter to return to main menu...")
        clear_screen()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        sys.exit(0)
