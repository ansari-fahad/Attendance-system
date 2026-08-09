# Face Recognition Attendance System — Pure OpenCV (LBPH) Version

**No TensorFlow dependency.** Works on any Python version, including 3.14 — good if
you can't easily set up a separate Python 3.12 environment right now.

## Trade-off vs the TensorFlow/DeepFace version
LBPH is lighter and dependency-free, but sensitive to lighting/angle/location changes
between enrollment and recognition. If you need reliable cross-location accuracy,
use the `attendance_tensorflow_deepface` version instead (needs Python 3.10-3.13).

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python enroll.py --id STU-001 --name "Priya Shah" --type student --department "CS-A" --webcam
python train_model.py
python mark_attendance.py
```

## Tuning
`config.py` → `CONFIDENCE_THRESHOLD` (default 50.0, LBPH distance, **lower = stricter**).
`SAMPLES_PER_PERSON` (default 30) — capture variety in angle/lighting for better generalization.

## Structure
```
attendance_opencv_lbph/
├── config.py
├── enroll.py
├── train_model.py
├── mark_attendance.py
├── core/
│   ├── face_detector.py     # Haar Cascade detection
│   └── recognizer.py        # LBPH matching + identity resolution
├── database/
│   └── db_manager.py        # SQLite: persons + attendance
├── data/known_faces/<id>/   # captured face samples
└── models/                  # lbph_model.yml + labels.pkl (from train_model.py)
```
