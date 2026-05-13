# convert_video.py
# Run this FIRST to convert your video
# Usage: python3 convert_video.py
# Edit the input_video name below to match your file

import cv2
import os

input_video  = "car1.mp4"   # ✏️ Change this to your video filename
output_video = "output.mp4"

if not os.path.exists(input_video):
    print(f"❌ '{input_video}' not found in this folder.")
    print(f"   Files here: {os.listdir('.')}")
    exit()

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"❌ Cannot open '{input_video}'.")
    exit()

fps   = cap.get(cv2.CAP_PROP_FPS) or 25
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 {w}x{h} @ {fps:.1f}fps | {total} frames")
print(f"🔄 Converting...")

out = cv2.VideoWriter(output_video,
      cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1
    if count % 100 == 0:
        print(f"   {count}/{total} frames done...")

cap.release()
out.release()
print(f"\n✅ Saved as '{output_video}'. Now run: streamlit run app.py")