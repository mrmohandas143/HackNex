"""
Downloads YOLOv4-tiny model files needed for cv2.dnn inference.
  - yolov4-tiny.cfg    (~5 KB  from AlexeyAB darknet)
  - yolov4-tiny.weights (~24 MB from AlexeyAB darknet)
  - coco.names          (~1 KB  COCO 80-class labels)
"""
import urllib.request, os, sys

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = [
    (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        os.path.join(MODELS_DIR, "yolov4-tiny.cfg"),
        5_000
    ),
    (
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        os.path.join(MODELS_DIR, "yolov4-tiny.weights"),
        10_000_000
    ),
    (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        os.path.join(MODELS_DIR, "coco.names"),
        500
    ),
]

proxy  = urllib.request.ProxyHandler()
opener = urllib.request.build_opener(proxy)
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
urllib.request.install_opener(opener)

all_ok = True
for url, dest, min_bytes in FILES:
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        print(f"[SKIP] {dest} already exists ({os.path.getsize(dest):,} bytes)")
        continue
    print(f"[DOWN] {os.path.basename(dest)}  <=  {url}")
    try:
        def progress(block_num, block_size, total):
            done = min(block_num * block_size, total)
            pct  = done / total * 100 if total > 0 else 0
            print(f"\r  {pct:5.1f}%  {done:>10,} / {total:>10,} bytes", end="", flush=True)
        urllib.request.urlretrieve(url, dest, progress)
        print()
        size = os.path.getsize(dest)
        if size >= min_bytes:
            print(f"[OK]   {dest}  ({size:,} bytes)")
        else:
            print(f"[FAIL] {dest} too small ({size} bytes)!")
            all_ok = False
    except Exception as e:
        print(f"[ERR]  {e}")
        all_ok = False

sys.exit(0 if all_ok else 1)
