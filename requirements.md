# System Requirements Specification Document
## Face Recognition Attendance Management System (AMS)

---

### 1. Executive Summary & Client Context
This document specifies the system requirements for a local, offline Face Recognition Attendance Management System (AMS) tailored to meet the client's specifications. 

#### Core Parameters:
*   **Database Capacity:** 5,000 to 10,000 registered students/employees (scalable for future growth).
*   **Administrative Access:** Approximately 10 dashboard/management users.
*   **Peak Volume Capacity:** 50 to 100 check-ins in a 15 to 30-minute interval without delays.
*   **Offline Setup:** 100% local deployment. All model inference, database tasks, and desktop GUIs must run without internet access.
*   **Camera Configuration:** Initial phase supports one input camera (standard USB webcam for development, IP CCTV camera stream for production), designed for multi-camera scalability in future updates.
*   **Physical Control:** AMS only. No integration with physical doors, gates, or barriers.

---

### 2. OpenCV LBPH (v1) Version Requirements

The OpenCV version implements Local Binary Patterns Histograms (LBPH) for face recognition. This is a lightweight, classical ML setup that runs on basic CPU cores.

#### Development Environment (Developer's Machine)
*   **CPU:** Intel Core i3 (6th Gen) or AMD Ryzen 3 (2 cores, 4 threads, 2.0 GHz) minimum.
*   **RAM:** 8 GB DDR4.
*   **GPU:** Integrated graphics only. No dedicated graphics card or CUDA setup is needed.
*   **Storage:** 2 GB available SSD space.
*   **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS.
*   **Python Support:** Python 3.7 to 3.14 (fully compatible).
*   **Core Libraries:** `opencv-contrib-python >= 4.8.0`, `numpy >= 1.24.0`, built-in `sqlite3`.

#### Production Environment (Client's Deployment PC)
*   **Host Machine:** Intel Core i5 or AMD Ryzen 5 PC, or a single-board computer like a Raspberry Pi 4 / 5 (8 GB RAM).
*   **RAM:** 8 GB RAM.
*   **Storage:** 128 GB SSD with at least 5 GB of free space (database files, config logs, cropped facial assets).
*   **Camera Infrastructure:** Standard 1080p USB webcam or IP CCTV camera supporting RTSP protocol (H.264).
*   **Operating System:** Windows 10/11 Pro (64-bit) or Ubuntu Linux 22.04 LTS.

> [!WARNING]
> **LBPH Scalability Warning:** Although LBPH has exceptionally low hardware costs, it is **not recommended** for the target volume of 5,000–10,000 users. Its accuracy degrades rapidly at this scale, leading to high false-match rates and recognition bottlenecks.

---
