# Vision AI — Tier 1 (Lightweight, Open-Source & Torch-Free)

A fast, lightweight, and completely local computer vision application that detects and tracks objects in real-time video (live webcams or pre-recorded videos) and answers natural language queries about the scene.

- **100% Torch-Free & Lightweight**: No heavy 2.5GB PyTorch download. Runs inference via OpenCV (`cv2.dnn`) and NumPy.
- **Fully Offline & Free**: Requires zero servers, zero API keys, and no paid resources.
- **Object Detection & Tracking**: Uses YOLOv4-tiny (~24MB) with a pure-NumPy IoU tracker assigning persistent track IDs across frames.
- **Multi-Camera Support**: Automatic detection of external USB webcams with runtime camera switching (`[C]`).
- **Scene Analysis & Q&A**: Answers questions about object counts, locations, presence, and colors (including clothing and t-shirt colors).

---

## Project Structure

```
HackNex/
│
├── models/                     # YOLOv4-tiny weights, cfg & COCO names (~24MB total)
│   ├── coco.names              # 80 COCO class labels
│   ├── yolov4-tiny.cfg         # Model network configuration
│   └── yolov4-tiny.weights     # Model weights (auto-downloadable)
│
├── detector.py                 # YOLOv4-tiny inference (cv2.dnn) + NumPy IoU tracker
├── scene_builder.py            # Extracts spatial positions & torso/clothing colors
├── qa_engine.py                # Rule-based natural language Q&A engine
├── utils.py                    # HSV color space analyzer & UI overlay drawing
├── setup_model.py              # Helper to download model weights if needed
├── main.py                     # Main application entry point
├── requirements.txt            # Minimal Python dependencies
└── .gitignore                  # Prevents venv, cache, and captures from being committed
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd HackNex
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights (if not already present)
```bash
python setup_model.py
```

---

## How to Run

### 1. Run with Webcam (Auto-Detects External Webcams)
```bash
venv\Scripts\python main.py
```
- If an external USB webcam is connected, `main.py` automatically detects and connects to it using DirectShow (`cv2.CAP_DSHOW`).
- You can also explicitly specify a camera index:
```bash
venv\Scripts\python main.py 0    # Built-in laptop webcam
venv\Scripts\python main.py 1    # External USB webcam
```

### 2. Run with a Pre-Recorded Video File
```bash
venv\Scripts\python main.py path\to\video.mp4
```

---

## Keyboard Controls (in Video Window)

| Key | Action |
|:---:|---|
| `[C]` | **Switch Camera** — Dynamically toggle between built-in and external webcams |
| `[Q]` | **Pause & Query** — Pauses the video and enters Q&A mode in the terminal |
| `[S]` | **Save Screenshot** — Saves the annotated frame as `capture_XXXXXX.png` |
| `[ESC]` | **Exit** — Closes the application |

---

## Interactive Q&A Mode `[Q]`

When you press `[Q]`, the stream pauses and captures the current frame. You can ask questions in natural language:

- **Scene Description**:
  - *"What is in the image?"* / *"What is in the frame?"* / *"Describe the scene"*
- **Clothing / T-shirt Color**:
  - *"Which color is the tshirt?"* / *"What color is the shirt?"* / *"What is the person wearing?"*
- **Object Counting**:
  - *"How many people are there?"* / *"How many cars are visible?"*
- **Location**:
  - *"Where is the person?"* / *"Where is the dog?"*
- **Presence Verification**:
  - *"Is there a cat?"* / *"Do you see any bicycles?"*

*Press **Enter** (blank query) or type `resume` to unpause the video and continue streaming.*
