# draw_slots.py
# Run this to mark parking slots by clicking and dragging on your video
# Usage: python3 draw_slots.py
# A window opens — draw rectangles over each parking slot, then press S to save

import cv2
import json
import os

# ✏️ CHANGE THIS to your converted video file name
VIDEO_FILE = "output.mp4"

# ── State ─────────────────────────────────────────────────────────────────
slots      = {}
slot_id    = 0
drawing    = False
start_x    = -1
start_y    = -1
base_frame = None

def draw_all(frame):
    for sid, s in slots.items():
        cv2.rectangle(frame, (s["x1"], s["y1"]), (s["x2"], s["y2"]), (0, 200, 80), 2)
        cv2.putText(frame, sid, (s["x1"] + 4, s["y1"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

def mouse(event, x, y, flags, param):
    global slot_id, drawing, start_x, start_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = base_frame.copy()
        draw_all(tmp)
        cv2.rectangle(tmp, (start_x, start_y), (x, y), (0, 200, 80), 2)
        cv2.imshow("Draw Slots", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx1, rx2 = min(start_x, x), max(start_x, x)
        ry1, ry2 = min(start_y, y), max(start_y, y)
        if abs(rx2 - rx1) > 10 and abs(ry2 - ry1) > 10:
            sid = f"slot_{slot_id}"
            slots[sid] = {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2, "occupied": False}
            slot_id += 1
            print(f"✅ Added {sid}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if slots:
            last = list(slots.keys())[-1]
            del slots[last]
            slot_id -= 1
            print(f"🗑  Removed {last}")

# ── Load first frame ──────────────────────────────────────────────────────
if not os.path.exists(VIDEO_FILE):
    print(f"❌ '{VIDEO_FILE}' not found. Run convert_video.py first.")
    exit()

cap = cv2.VideoCapture(VIDEO_FILE)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame from video.")
    exit()

# Resize if too wide
h, w = frame.shape[:2]
if w > 1100:
    scale = 1100 / w
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

base_frame = frame.copy()

# ── Open window ───────────────────────────────────────────────────────────
cv2.namedWindow("Draw Slots")
cv2.setMouseCallback("Draw Slots", mouse)

print("\n🖱  LEFT CLICK + DRAG  → draw a parking slot rectangle")
print("🖱  RIGHT CLICK        → remove last slot")
print("⌨   S                  → save and quit")
print("⌨   Q                  → quit without saving\n")

while True:
    display = base_frame.copy()

    # HUD bar
    cv2.rectangle(display, (0, 0), (display.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(display,
                f"Slots: {len(slots)}   |   LEFT DRAG = draw   RIGHT CLICK = undo   S = save   Q = quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 180), 1, cv2.LINE_AA)

    draw_all(display)
    cv2.imshow("Draw Slots", display)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('s'):
        with open("slots.json", "w") as f:
            json.dump(slots, f, indent=2)
        print(f"\n💾 Saved {len(slots)} slots to slots.json")
        print("   Now run:  streamlit run app.py")
        break
    elif key == ord('q'):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()