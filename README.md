# Vision AI — Tier 2 (Deep Attribute Inspection, Temporal Tracking & Conversational Q&A)

A fast, lightweight, and completely local computer vision application that detects and tracks objects in real-time video (live webcams or pre-recorded videos) and performs deep multimodal visual reasoning without external APIs.

- **100% Torch-Free & Lightweight**: No heavy 2.5GB PyTorch download. Runs inference via OpenCV (`cv2.dnn`), NumPy, and ONNX Runtime.
- **Fully Offline & Free**: Requires zero servers, zero API keys, and no paid resources.
- **Deep Anatomical & Attribute Inspection**: Sub-region analysis for earrings, eyeglasses, headwear, upper/lower clothing colors, and posture (standing vs. sitting).
- **Weapons & Carried Items Detection**: Perimeter and hand-zone inspection for rigid elongated objects or carried items.
- **Temporal Dynamics & Motion History**: Real-time velocity tracking, motion direction (moving left/right/approaching/stationary), dwell times, and entry/exit events.
- **Multi-Turn Conversational Memory**: Understands context, pronouns, and follow-up fragments (*"And what about the pants?"*).
- **Multi-Camera Support**: Automatic detection of external USB webcams with DirectShow and runtime camera switching (`[C]`).

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
├── attribute_analyzer.py       # Tier 2: Anatomical zones (earrings, glasses, weapons, posture)
├── temporal_tracker.py         # Tier 2: Motion vectors, dwell time, entry/exit event logs
├── conversation_engine.py      # Tier 2: Multi-turn memory, follow-ups & coreference resolution
├── scene_builder.py            # Enriches detections with attributes & temporal metrics
├── qa_engine.py                # Visual reasoning & conversational natural language Q&A
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
git clone https://github.com/mrmohandas143/HackNex.git
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
| `[Q]` | **Pause & Query** — Pauses the video and enters interactive Q&A mode in the terminal |
| `[S]` | **Save Screenshot** — Saves the annotated frame as `capture_XXXXXX.png` |
| `[ESC]` | **Exit** — Closes the application |

---

## Example Queries in Tier 2 Q&A Mode `[Q]`

When you press `[Q]`, the stream pauses and captures the current frame. You can ask open-ended questions in natural language:

### Accessories & Fine-Grained Attributes
- *"Does the person wear any earrings?"*
- *"Is the person wearing glasses?"*
- *"Is the person wearing a hat?"*

### Weapons & Carried Items
- *"Does the person have any weapon?"*
- *"Are they holding anything?"*

### Posture & Actions
- *"Is the person sitting or standing?"*
- *"Which direction are they moving?"*
- *"Is anyone moving or are they still?"*

### Clothing & Outfits
- *"What is the person wearing?"* (summarizes shirt, pants, accessories)
- *"Which color is the tshirt?"*
- *"What color are the pants?"*

### Temporal & Activity Metrics
- *"How long has the person been here?"*
- *"Did anyone enter or leave recently?"*

### Conversational Follow-Ups & Memory
- **You**: *"What color is the shirt?"* ➔ **AI**: *"The person's shirt appears to be green."*
- **You**: *"And what about the pants?"* ➔ **AI**: *"The person's pants appear to be blue."*
- **You**: *"Repeat that"* ➔ **AI**: *"Previously, you asked..."*

*Press **Enter** (blank query) or type `resume` to unpause the video and continue streaming.*
