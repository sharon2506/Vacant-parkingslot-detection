import streamlit as st
import cv2
import json
import numpy as np
import tempfile
import time
import base64
import io
import math
from datetime import datetime

st.set_page_config(page_title="ParkVision Pro", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS — EXACTLY THE ORIGINAL ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main,.main .block-container{
    background:#f4f6fb !important;font-family:'Inter',sans-serif !important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;}
.block-container{padding:1.4rem 2rem !important;max-width:1300px;}

.topbar{background:linear-gradient(135deg,#0a2463,#1565c0,#1976d2);
    border-radius:18px;padding:1.3rem 1.8rem;display:flex;align-items:center;
    gap:1rem;margin-bottom:1.4rem;box-shadow:0 6px 24px rgba(10,36,99,.35);}
.topbar-icon{font-size:2.4rem;}
.topbar-title{font-size:1.65rem;font-weight:800;color:#fff;margin:0;letter-spacing:-.02em;}
.topbar-sub{font-size:.79rem;color:#90caf9;margin:.2rem 0 0;}
.live-pill{margin-left:auto;background:rgba(255,80,80,.85);border-radius:999px;
    padding:.28rem 1rem;font-size:.72rem;color:#fff;font-weight:700;
    animation:pulse 1.5s infinite;white-space:nowrap;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.6;}}

.stepbar{display:flex;align-items:center;gap:.4rem;margin-bottom:1.4rem;}
.stp{display:flex;align-items:center;gap:.4rem;font-size:.78rem;font-weight:600;color:#bdbdbd;}
.stp.active{color:#1565c0;}.stp.done{color:#2e7d32;}
.stp-n{width:26px;height:26px;border-radius:50%;font-size:.72rem;font-weight:700;
    display:flex;align-items:center;justify-content:center;background:#e0e0e0;color:#9e9e9e;}
.stp.active .stp-n{background:#1565c0;color:#fff;box-shadow:0 2px 8px rgba(21,101,192,.4);}
.stp.done .stp-n{background:#2e7d32;color:#fff;}
.stp-line{flex:1;height:2px;background:#e0e0e0;border-radius:2px;}

.card{background:#fff;border:1.5px solid #e3eaf7;border-radius:14px;
    padding:1.2rem 1.3rem;margin-bottom:.9rem;box-shadow:0 2px 10px rgba(0,0,0,.05);}
.card-title{font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
    color:#1565c0;margin-bottom:.8rem;}

.srow{display:flex;gap:.75rem;margin-bottom:.9rem;}
.sc{flex:1;border-radius:14px;padding:1.1rem .7rem;text-align:center;}
.sc-t{background:linear-gradient(135deg,#e8eaf6,#c5cae9);border:1.5px solid #9fa8da;}
.sc-v{background:linear-gradient(135deg,#e8f5e9,#a5d6a7);border:1.5px solid #81c784;}
.sc-o{background:linear-gradient(135deg,#ffebee,#ef9a9a);border:1.5px solid #e57373;}
.sc-c{background:linear-gradient(135deg,#fff8e1,#ffe082);border:1.5px solid #ffd54f;}
.sc-n{font-size:2.8rem;font-weight:800;line-height:1;}
.sc-t .sc-n{color:#283593;}.sc-v .sc-n{color:#1b5e20;}
.sc-o .sc-n{color:#b71c1c;}.sc-c .sc-n{color:#e65100;}
.sc-l{font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem;}
.sc-t .sc-l{color:#5c6bc0;}.sc-v .sc-l{color:#388e3c;}
.sc-o .sc-l{color:#e53935;}.sc-c .sc-l{color:#f57f17;}

.pw{background:#e0e0e0;border-radius:999px;height:13px;overflow:hidden;margin:.7rem 0 .35rem;}
.pf{height:100%;border-radius:999px;transition:width .5s ease;}
.pl{display:flex;justify-content:space-between;font-size:.71rem;color:#9e9e9e;font-weight:500;}

.ibox{background:#e3f2fd;border-left:4px solid #1565c0;border-radius:0 10px 10px 0;
    padding:.8rem 1rem;font-size:.82rem;color:#0d47a1;margin-bottom:.8rem;line-height:1.8;}
.wbox{background:#fff8e1;border-left:4px solid #ffa000;border-radius:0 10px 10px 0;
    padding:.8rem 1rem;font-size:.82rem;color:#e65100;margin-bottom:.8rem;line-height:1.8;}
.sbox{background:#e8f5e9;border-left:4px solid #43a047;border-radius:0 10px 10px 0;
    padding:.8rem 1rem;font-size:.82rem;color:#1b5e20;margin-bottom:.8rem;line-height:1.8;}

.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color:#fff !important;border:none !important;border-radius:10px !important;
    font-family:'Inter',sans-serif !important;font-weight:600 !important;
    font-size:.85rem !important;width:100% !important;padding:.6rem 1.2rem !important;}
.stButton>button:hover{opacity:.87 !important;}

.ev-box{background:#f8f9ff;border:1px solid #e3eaf7;border-radius:10px;
    padding:.75rem 1rem;font-size:.76rem;color:#424242;line-height:2;
    max-height:130px;overflow-y:auto;margin-top:.5rem;}

.sum-wrap{background:#fff;border:2px solid #c5cae9;border-radius:20px;
    padding:2rem;margin-top:.5rem;box-shadow:0 4px 20px rgba(0,0,0,.08);}
.sum-header{text-align:center;margin-bottom:1.5rem;}
.sum-icon{font-size:3.5rem;}
.sum-title{font-size:1.5rem;font-weight:800;color:#0d47a1;margin:.3rem 0 .2rem;}
.sum-sub{font-size:.85rem;color:#78909c;}
.sum-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1.2rem;}
.sm{border-radius:14px;padding:1rem .7rem;text-align:center;}
.sm-t{background:#e8eaf6;}.sm-v{background:#e8f5e9;}
.sm-o{background:#ffebee;}.sm-c{background:#fff8e1;}
.sm .n{font-size:2.4rem;font-weight:800;line-height:1;}
.sm-t .n{color:#283593;}.sm-v .n{color:#1b5e20;}
.sm-o .n{color:#b71c1c;}.sm-c .n{color:#e65100;}
.sm .l{font-size:.65rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;margin-top:.2rem;}
.sm-t .l{color:#5c6bc0;}.sm-v .l{color:#388e3c;}
.sm-o .l{color:#e53935;}.sm-c .l{color:#f57f17;}
.sum-bar-wrap{background:#e0e0e0;border-radius:999px;height:14px;
    overflow:hidden;margin:.8rem 0 .35rem;}
.sum-bar{height:100%;border-radius:999px;}
.sum-status{text-align:center;font-size:1rem;font-weight:700;
    padding:.7rem;border-radius:10px;margin-top:.8rem;}

.splash-wrap{background:linear-gradient(135deg,#0a2463,#1565c0,#1976d2);
    border-radius:24px;padding:3rem 2rem;text-align:center;
    box-shadow:0 12px 40px rgba(10,36,99,.4);margin:1rem 0;}
.splash-icon{font-size:5rem;margin-bottom:.8rem;}
.splash-title{font-size:2.4rem;font-weight:800;color:#fff;margin-bottom:.4rem;}
.splash-sub{font-size:.98rem;color:#90caf9;margin-bottom:1.8rem;line-height:1.7;}
.feat-row{display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.8rem;}
.feat{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);
    border-radius:999px;padding:.38rem 1rem;font-size:.77rem;color:#fff;font-weight:600;}

.alert-full{background:#ffebee;border:2px solid #e53935;border-radius:10px;
    padding:.65rem 1rem;font-size:.82rem;color:#b71c1c;font-weight:700;
    text-align:center;animation:alertp 1.6s infinite;}
@keyframes alertp{0%,100%{box-shadow:0 0 0 0 rgba(229,57,53,.3);}
    50%{box-shadow:0 0 0 8px rgba(229,57,53,0);}}
.alert-ok{background:#e8f5e9;border:2px solid #43a047;border-radius:10px;
    padding:.65rem 1rem;font-size:.82rem;color:#1b5e20;font-weight:700;text-align:center;}

.an-row{display:flex;gap:.6rem;margin:.4rem 0;}
.an-chip{flex:1;background:#f8f9ff;border:1px solid #e3eaf7;border-radius:10px;
    padding:.55rem .4rem;text-align:center;}
.an-chip .n{font-size:1.25rem;font-weight:800;color:#1565c0;}
.an-chip .l{font-size:.58rem;color:#9e9e9e;text-transform:uppercase;
    letter-spacing:.07em;font-weight:600;margin-top:.1rem;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(
    slots={}, video_path=None, first_frame=None,
    ref_frame=None, stage="splash", scale=1.0,
    show_summary=False, summary={}, calib_done=False,
    events=[], slot_entry_times={}, dwell_times=[],
    occ_history=[], max_cars_seen=0, session_start=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# URL param slot receiver
qp = st.query_params
if "pvs" in qp and st.session_state.stage == "draw":
    try:
        raw = json.loads(qp["pvs"])
        zones = {f"zone_{i}": {
            "x1": int(float(r["x1"])), "y1": int(float(r["y1"])),
            "x2": int(float(r["x2"])), "y2": int(float(r["y2"]))}
            for i, r in enumerate(raw)}
        if zones:
            st.session_state.slots = zones
            st.session_state.stage = "calibrate"
            st.query_params.clear()
            st.rerun()
    except:
        pass


# ── UI helpers ────────────────────────────────────────────────────────────────
def stepbar(cur):
    labels = ["Upload", "Draw Slots", "Calibrate", "Detect"]
    h = '<div class="stepbar">'
    for i, l in enumerate(labels, 1):
        cls  = "done"   if i < cur else ("active" if i == cur else "stp")
        icon = "✓"      if i < cur else str(i)
        h += f'<div class="stp {cls}"><div class="stp-n">{icon}</div>{l}</div>'
        if i < len(labels):
            h += '<div class="stp-line"></div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)


def show_topbar(live=False):
    pill = '<div class="live-pill">● LIVE</div>' if live else ''
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-icon">🚗</div>
      <div>
        <p class="topbar-title">ParkVision Pro</p>
        <p class="topbar-sub">Real-time parking slot detector · YOLOv8-Primary + Smart BG</p>
      </div>
      {pill}
    </div>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8s.pt")


def small(frame, w=460):
    h, fw = frame.shape[:2]
    if fw > w:
        frame = cv2.resize(frame, (w, int(h * w / fw)))
    return frame


def to_b64(frame, max_w=680):
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode(), frame.shape[1], frame.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
# SLOT DETECTION FROM DRAWN ZONE
# ══════════════════════════════════════════════════════════════════════════════

def find_peaks(signal, dim, min_dist_frac=0.06):
    if len(signal) == 0:
        return []
    mean = signal.mean()
    std  = signal.std()
    thr  = mean + 0.7 * std
    min_d = max(10, int(dim * min_dist_frac))
    peaks = []
    for i in range(min_d, len(signal) - min_d):
        if signal[i] > thr:
            window = signal[max(0, i - min_d): i + min_d]
            if signal[i] == window.max():
                peaks.append(i)
    return peaks


def make_slot(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "occupied": False, "score": 0.0, "ema": 0.0}


def detect_slots_in_zone(frame, zone, scale):
    """
    Detect individual parking slots inside a drawn zone.
    FIX BUG 3: Improved fallback for thin zones, and minimum slot size enforced.
    """
    sx = 1.0 / scale

    zx1 = max(0, int(zone["x1"] * sx))
    zy1 = max(0, int(zone["y1"] * sx))
    zx2 = min(frame.shape[1], int(zone["x2"] * sx))
    zy2 = min(frame.shape[0], int(zone["y2"] * sx))
    zw  = zx2 - zx1
    zh  = zy2 - zy1

    MIN_SLOT_PX = 20  # FIX: don't create slivers smaller than this

    if zw < 10 or zh < 10:
        return [make_slot(zx1, zy1, zx2, zy2)]

    roi  = frame[zy1:zy2, zx1:zx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    sobelx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_proj = np.abs(sobelx).mean(axis=0)
    col_smooth = np.convolve(col_proj,
                              np.ones(max(3, zw // 30)) / max(3, zw // 30),
                              mode='same')
    v_peaks = find_peaks(col_smooth, zw, min_dist_frac=0.06)

    sobely   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_proj = np.abs(sobely).mean(axis=1)
    row_smooth = np.convolve(row_proj,
                              np.ones(max(3, zh // 20)) / max(3, zh // 20),
                              mode='same')
    h_peaks = find_peaks(row_smooth, zh, min_dist_frac=0.08)

    slots = []

    if len(v_peaks) >= len(h_peaks) and len(v_peaks) > 0:
        xs = sorted([0] + v_peaks + [zw])
        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] < MIN_SLOT_PX:  # FIX: skip slivers
                continue
            slots.append(make_slot(zx1 + xs[i], zy1, zx1 + xs[i + 1], zy2))
    elif len(h_peaks) > 0:
        ys = sorted([0] + h_peaks + [zh])
        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] < MIN_SLOT_PX:  # FIX: skip slivers
                continue
            slots.append(make_slot(zx1, zy1 + ys[i], zx2, zy1 + ys[i + 1]))

    # FIX BUG 3 — Improved fallback heuristic
    if not slots:
        aspect = zw / max(zh, 1)
        if aspect >= 1.6:
            # Wide zone → split horizontally into slots
            n = max(1, round(aspect / 1.5))
            sw = zw // n
            if sw >= MIN_SLOT_PX:
                for i in range(n):
                    x1s = zx1 + i * sw
                    x2s = zx1 + (i + 1) * sw if i < n - 1 else zx2
                    slots.append(make_slot(x1s, zy1, x2s, zy2))
        elif aspect <= 0.6:
            # Tall zone → split vertically into slots
            n = max(1, round((1 / aspect) / 1.5))
            sh = zh // n
            if sh >= MIN_SLOT_PX:
                for i in range(n):
                    y1s = zy1 + i * sh
                    y2s = zy1 + (i + 1) * sh if i < n - 1 else zy2
                    slots.append(make_slot(zx1, y1s, zx2, y2s))

        if not slots:
            slots.append(make_slot(zx1, zy1, zx2, zy2))

    return slots if slots else [make_slot(zx1, zy1, zx2, zy2)]


def zones_to_slots(zones_dict, frame, scale):
    all_slots = {}
    idx = 0
    for zid, zone in zones_dict.items():
        for s in detect_slots_in_zone(frame, zone, scale):
            all_slots[f"slot_{idx}"] = s
            idx += 1
    return all_slots


# ══════════════════════════════════════════════════════════════════════════════
# SMART REFERENCE BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build_smart_reference(frames_gray_list):
    stack = np.stack(frames_gray_list, axis=0).astype(np.float32)
    ref_max = np.max(stack, axis=0).astype(np.uint8)
    ref_min = np.min(stack, axis=0).astype(np.uint8)
    return ref_max, ref_min


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION FUNCTIONS — FIXED
# ══════════════════════════════════════════════════════════════════════════════

def check_slot_bg(frame_gray, ref_max, ref_min, slot):
    """BG subtraction — SECONDARY detector only. Same logic but thresholds tuned."""
    x1 = max(0, slot["x1"]); y1 = max(0, slot["y1"])
    x2 = min(frame_gray.shape[1], slot["x2"])
    y2 = min(frame_gray.shape[0], slot["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    cur  = cv2.GaussianBlur(frame_gray[y1:y2, x1:x2], (5,5), 0).astype(np.float32)
    rmax = cv2.GaussianBlur(ref_max[y1:y2, x1:x2], (5,5), 0).astype(np.float32)
    rmin = cv2.GaussianBlur(ref_min[y1:y2, x1:x2], (5,5), 0).astype(np.float32)

    dark_diff  = np.clip(rmax - cur, 0, 255)
    dark_frac  = (dark_diff > 25).sum() / max(1, dark_diff.size)
    bright_diff = np.clip(cur - rmin, 0, 255)
    bright_frac = (bright_diff > 30).sum() / max(1, bright_diff.size)
    var_cur = float(np.std(cur))
    var_ref = float(np.std(rmax))
    var_score = min(abs(var_cur - var_ref) / max(var_ref + 1, 1), 1.0)

    score = 0.50 * dark_frac + 0.30 * bright_frac + 0.20 * var_score
    return float(score)


def check_slot_yolo(slot, boxes, fw, fh):
    """
    FIX BUG 1 — Use proper IoU overlap, not just center-point check.
    A vehicle overlapping ≥8% of the slot area → OCCUPIED.
    Also check if slot is mostly inside a large vehicle box.
    """
    x1 = max(0, slot["x1"]); y1 = max(0, slot["y1"])
    x2 = min(fw, slot["x2"]); y2 = min(fh, slot["y2"])
    slot_area = max(1, (x2 - x1) * (y2 - y1))
    best_overlap = 0.0
    best_iou = 0.0

    for box in boxes:
        bx1, by1, bx2, by2 = [float(v) for v in box[:4]]

        # Intersection area
        ix1 = max(bx1, x1); iy1 = max(by1, y1)
        ix2 = min(bx2, x2); iy2 = min(by2, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        if inter <= 0:
            continue

        # Overlap as fraction of slot area
        frac_of_slot = inter / slot_area

        # Also check fraction of vehicle area that's in slot (catches large vehicles)
        vehicle_area = max(1, (bx2 - bx1) * (by2 - by1))
        frac_of_vehicle = inter / vehicle_area

        # Union for IoU
        union = slot_area + vehicle_area - inter
        iou = inter / max(union, 1)

        combined = max(frac_of_slot, frac_of_vehicle * 0.5, iou)
        if combined > best_overlap:
            best_overlap = combined
            best_iou = iou

    # FIX: threshold lowered to 0.08 (8%) — catches partial overlaps
    is_occupied = best_overlap >= 0.08
    return is_occupied, best_overlap


VOTE_WINDOW = 8  # FIX: reduced from 10 for faster response


def decide_occupied(yolo_occ, yolo_overlap, bg_score, slot):
    """
    FIX: Symmetric vote thresholds + active False-push when YOLO sees nothing.

    Key insight: when YOLO finds zero overlap (no vehicle detected near slot at all)
    AND the BG score is weak, we push TWO Falses into the buffer. This is how stuck
    'permanently OCCUPIED' slots get unstuck — previously they just accumulated weak
    Trues from BG and could never recover.
    """
    import collections
    if "vote_buf" not in slot:
        slot["vote_buf"] = collections.deque(maxlen=VOTE_WINDOW)

    # High-confidence YOLO → instantly occupied, flood the buffer
    if yolo_overlap >= 0.35:
        slot["vote_buf"].extend([True] * VOTE_WINDOW)
        return True

    # YOLO sees NOTHING near this slot AND BG is weak → actively push False
    # This releases slots stuck as OCCUPIED with no real vehicle
    if yolo_overlap == 0.0 and bg_score < 0.30:
        slot["vote_buf"].append(False)
        slot["vote_buf"].append(False)  # double-push for faster release

    # Raw frame decision
    if yolo_occ and yolo_overlap >= 0.08:
        raw = True
    elif yolo_occ and bg_score >= 0.12:
        raw = True
    elif bg_score >= 0.65:
        raw = True   # Raised: only very strong BG alone → occupied
    elif bg_score >= 0.40 and slot.get("occupied", False) and yolo_overlap > 0.0:
        raw = True   # Medium BG + was occupied + YOLO partially agrees → keep
    else:
        raw = False

    slot["vote_buf"].append(raw)

    buf = list(slot["vote_buf"])
    if len(buf) < max(3, VOTE_WINDOW // 3):
        return slot.get("occupied", False)

    true_frac = sum(buf) / len(buf)
    prev = slot.get("occupied", False)

    if prev:
        return true_frac >= 0.40   # Was occupied: needs 60%+ False to go vacant
    else:
        return true_frac >= 0.50   # Was vacant: needs 50%+ True to go occupied


def detect_vehicles_yolo(frame, model):
    res   = model.predict(frame, classes=[2, 3, 5, 7],
                          conf=0.20, iou=0.40, verbose=False)
    boxes = []
    if res and res[0].boxes is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist()
    return boxes


def draw_result(frame, slots, boxes, blink_phase=0):
    """
    FIX BUG 4 — Vacant slots prominently highlighted with bright green glow.
    Occupied slots clearly marked RED. No more faint overlays.
    """
    fh, fw = frame.shape[:2]

    # Draw YOLO vehicle boxes (cyan/teal color for visibility)
    for box in boxes:
        bx1, by1, bx2, by2 = [int(v) for v in box[:4]]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 255), 2)
        cv2.putText(frame, "Car", (bx1, max(by1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 220, 255), 1, cv2.LINE_AA)

    # Draw slots
    for sid, slot in slots.items():
        x1 = max(0, slot["x1"]); y1 = max(0, slot["y1"])
        x2 = min(fw, slot["x2"]); y2 = min(fh, slot["y2"])
        occ = slot.get("occupied", False)
        n   = int(sid.split("_")[1]) + 1

        overlay = frame.copy()

        if occ:
            # ── OCCUPIED: red fill + bold border ─────────────────────────
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

            # Bold red border (3px)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)

            # "FULL" badge — top-left
            label = "FULL"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx2b = x1 + tw + 8
            by2b = y1 + th + 7
            cv2.rectangle(frame, (x1, y1), (bx2b, by2b), (0, 0, 180), -1)
            cv2.putText(frame, label, (x1 + 4, y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Slot number — bottom-left
            cv2.putText(frame, f"P{n}", (x1 + 3, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 200, 200), 1, cv2.LINE_AA)

        else:
            # ── VACANT: bright green glow + pulsing border ────────────────
            # Green fill overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 50), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

            # Pulsing green border — thick when blink_phase=0, thinner when 1
            border_thick = 3 if blink_phase % 2 == 0 else 2
            border_color = (0, 230, 70) if blink_phase % 2 == 0 else (0, 160, 40)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

            # Outer glow on blink_phase 0 (draw slightly larger rectangle)
            if blink_phase % 2 == 0:
                gx1 = max(0, x1 - 2); gy1 = max(0, y1 - 2)
                gx2 = min(fw, x2 + 2); gy2 = min(fh, y2 + 2)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 100), 1)

            # "FREE ✓" badge — top-left, bright green background
            label = "FREE"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx2b = x1 + tw + 8
            by2b = y1 + th + 7
            badge_color = (0, 190, 50) if blink_phase % 2 == 0 else (0, 130, 35)
            cv2.rectangle(frame, (x1, y1), (bx2b, by2b), badge_color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Slot number — bottom-left
            cv2.putText(frame, f"P{n}", (x1 + 3, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 255, 180), 1, cv2.LINE_AA)

            # Green dot indicator — bottom-right corner
            dot_x = max(x1 + 8, min(x2 - 8, x2 - 8))
            dot_y = max(y1 + 8, min(y2 - 8, y2 - 8))
            dot_r = max(4, min(8, (x2 - x1) // 6, (y2 - y1) // 6))
            dot_color = (0, 230, 70) if blink_phase % 2 == 0 else (0, 160, 40)
            cv2.circle(frame, (dot_x, dot_y), dot_r, dot_color, -1)
            cv2.circle(frame, (dot_x, dot_y), dot_r, (255, 255, 255), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def make_report(summary, events, dwell_times, session_start):
    total = summary.get("total", 0); free = summary.get("free", 0)
    occ   = summary.get("occ",   0); cars = summary.get("cars", 0)
    peak  = summary.get("peak",  0)
    pct   = int(occ / total * 100) if total else 0
    avg_d = f"{np.mean(dwell_times):.1f}s" if dwell_times else "N/A"
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        T = ParagraphStyle('T', fontSize=22, fontName='Helvetica-Bold',
                            textColor=colors.HexColor('#0a2463'),
                            spaceAfter=4, alignment=TA_CENTER)
        S = ParagraphStyle('S', fontSize=10,
                            textColor=colors.HexColor('#78909c'),
                            alignment=TA_CENTER, spaceAfter=12)
        H = ParagraphStyle('H', fontSize=13, fontName='Helvetica-Bold',
                            textColor=colors.HexColor('#1565c0'),
                            spaceBefore=14, spaceAfter=6)
        story = []
        story.append(Paragraph("ParkVision Pro", T))
        story.append(Paragraph("Parking Session Report", S))
        story.append(Paragraph(
            f"Generated: {now}  |  Session start: {session_start or 'N/A'}", S))
        story.append(HRFlowable(width="100%", thickness=2,
                                  color=colors.HexColor('#1565c0'), spaceAfter=12))
        story.append(Paragraph("Session Summary", H))
        data = [
            ["Metric", "Value"],
            ["Total Parking Slots",     str(total)],
            ["Vacant (session end)",    str(free)],
            ["Occupied (session end)",  str(occ)],
            ["Occupancy Rate",          f"{pct}%"],
            ["Peak Occupancy",          str(peak)],
            ["Vehicles Detected (YOLO)", str(cars)],
            ["Average Dwell Time",      avg_d],
            ["Total Events Logged",     str(len(events))],
        ]
        t = Table(data, colWidths=[10*cm, 7*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0),  colors.HexColor('#1565c0')),
            ('TEXTCOLOR',   (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE',    (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#f4f6fb'), colors.white]),
            ('GRID',        (0, 0), (-1, -1), .4, colors.HexColor('#e3eaf7')),
            ('TOPPADDING',  (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))
        if events:
            story.append(Paragraph("Event Log", H))
            ev_data = [["#", "Event"]] + [[str(i+1), e] for i, e in enumerate(events)]
            t2 = Table(ev_data, colWidths=[1.5*cm, 15.5*cm])
            t2.setStyle(TableStyle([
                ('BACKGROUND',  (0, 0), (-1, 0),  colors.HexColor('#e3eaf7')),
                ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',    (0, 0), (-1, -1), 9),
                ('ALIGN',       (0, 0), (0, -1),  'CENTER'),
                ('GRID',        (0, 0), (-1, -1), .3, colors.HexColor('#e3eaf7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor('#f9faff'), colors.white]),
                ('TOPPADDING',  (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1,
                                  color=colors.HexColor('#e3eaf7')))
        story.append(Paragraph(
            "ParkVision Pro · YOLOv8-Primary Detection · Fixed v2",
            ParagraphStyle('ft', fontSize=8, textColor=colors.HexColor('#9e9e9e'),
                           alignment=TA_CENTER, spaceBefore=8)))
        doc.build(story)
        buf.seek(0)
        return buf.read(), "application/pdf", "parkvision_report.pdf"

    except ImportError:
        rows = "".join(f"<tr><td>{i+1}</td><td>{e}</td></tr>"
                       for i, e in enumerate(events))
        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
        <title>ParkVision Report</title>
        <style>
          body{{font-family:Arial,sans-serif;margin:40px;color:#333;background:#f4f6fb;}}
          h1{{color:#0a2463;}} h2{{color:#1565c0;margin-top:24px;}}
          .grid{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0;}}
          .stat{{background:#fff;border:1.5px solid #e3eaf7;border-radius:12px;
              padding:14px 20px;text-align:center;min-width:110px;}}
          .stat .n{{font-size:2rem;font-weight:800;color:#283593;}}
          .stat .l{{font-size:.75rem;color:#5c6bc0;text-transform:uppercase;}}
          table{{border-collapse:collapse;width:100%;margin-top:8px;
              background:#fff;border-radius:10px;overflow:hidden;}}
          th{{background:#1565c0;color:#fff;padding:9px 12px;text-align:left;}}
          td{{padding:7px 12px;border-bottom:1px solid #e3eaf7;font-size:.88rem;}}
          tr:nth-child(even){{background:#f4f6fb;}}
        </style></head><body>
        <h1>ParkVision Pro — Session Report (Fixed v2)</h1>
        <p style="color:#78909c;">Generated: {now} | Session: {session_start or 'N/A'}</p>
        <div class="grid">
          <div class="stat"><div class="n">{total}</div><div class="l">Total Slots</div></div>
          <div class="stat"><div class="n" style="color:#1b5e20;">{free}</div>
            <div class="l">Vacant</div></div>
          <div class="stat"><div class="n" style="color:#b71c1c;">{occ}</div>
            <div class="l">Occupied</div></div>
          <div class="stat"><div class="n">{pct}%</div><div class="l">Occupancy</div></div>
          <div class="stat"><div class="n">{peak}</div><div class="l">Peak Occ.</div></div>
          <div class="stat"><div class="n">{cars}</div><div class="l">Cars Seen</div></div>
          <div class="stat"><div class="n">{avg_d}</div><div class="l">Avg Dwell</div></div>
        </div>
        <h2>Event Log ({len(events)} events)</h2>
        <table><tr><th>#</th><th>Event</th></tr>{rows}</table>
        </body></html>"""
        return html.encode(), "text/html", "parkvision_report.html"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — SPLASH
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "splash":
    st.markdown("""
    <div class="splash-wrap">
      <div class="splash-icon">🚗</div>
      <div class="splash-title">ParkVision Pro</div>
      <div class="splash-sub">
        Intelligent real-time parking space monitoring<br>
        YOLOv8-Primary · Smart Background Reference · Precise IoU Overlap
      </div>
      <div class="feat-row">
        <span class="feat">🎯 IoU-Based Slot Detection</span>
        <span class="feat">📊 Live Analytics</span>
        <span class="feat">💡 Glowing Vacant Highlights</span>
        <span class="feat">🚨 Lot-Full Alert</span>
        <span class="feat">📋 PDF Report</span>
        <span class="feat">⏱ Dwell Time Tracking</span>
      </div>
    </div>""", unsafe_allow_html=True)

    _, cc, _ = st.columns([1, 1, 1])
    with cc:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Start Parking Space Detection"):
            st.session_state.stage = "upload"
            st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
      <div class="card"><div class="card-title">Fixes in v2</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          ✅ IoU overlap (was center-point only)<br>
          ✅ YOLO is now primary detector<br>
          ✅ No more thin sliver slots<br>
          ✅ Bright green glow for vacant slots<br>
          ✅ Symmetric vote thresholds<br>
          ✅ High-conf YOLO bypasses vote
        </div></div>
      <div class="card"><div class="card-title">Detection Engine</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          🟡 YOLOv8 (PRIMARY — always runs)<br>
          🔵 IoU overlap ≥8% → OCCUPIED<br>
          🔵 Background subtraction (secondary)<br>
          🟢 Vote buffer (8 frames, symmetric)<br>
          🔴 High-conf YOLO bypasses buffer
        </div></div>
      <div class="card"><div class="card-title">What You Get</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          🟢 Glowing green = VACANT (pulsing)<br>
          🔴 Bold red = OCCUPIED<br>
          📊 Total / Vacant / Occupied live<br>
          🔔 Lot-full alert at 95%<br>
          📥 Download PDF or HTML report
        </div></div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "upload":
    show_topbar()
    stepbar(1)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown('<div class="card"><div class="card-title">Upload Parking Lot Video</div>',
                    unsafe_allow_html=True)
        up = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"],
                               label_visibility="collapsed")
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read()); tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            ret, frame = cap.read(); cap.release()
            if ret and frame is not None:
                st.session_state.video_path  = tmp.name
                st.session_state.first_frame = frame.copy()
                st.image(cv2.cvtColor(small(frame, 480), cv2.COLOR_BGR2RGB),
                         caption="✅ Video loaded", use_column_width=False)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➡️  Next: Draw Slots"):
                    st.session_state.stage = "draw"; st.rerun()
            else:
                st.error("❌ Can't read video. Try converting first.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="ibox"><b>🎯 v2 Detection approach</b><br><br>
        <b>YOLO is PRIMARY:</b><br>
        1️⃣ YOLOv8 detects every vehicle (conf ≥ 20%)<br>
        2️⃣ IoU overlap ≥ 8% with slot → OCCUPIED<br>
        3️⃣ High IoU (≥35%) → instantly occupied<br>
        4️⃣ Background subtraction as fallback<br>
        5️⃣ Vote buffer prevents flicker<br><br>
        <b>Vacant slots glow bright green 🟢</b>
        </div>
        <div class="wbox"><b>Drawing tips</b><br>
        Draw boxes tightly around each parking row.<br>
        Make sure cars would be <i>inside</i> your box.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DRAW SLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "draw":
    show_topbar()
    stepbar(2)

    frame = st.session_state.first_frame
    CW = 620; h0, w0 = frame.shape[:2]
    sc = CW / w0; ch = int(h0 * sc)
    st.session_state.scale = sc
    b64, cw, _ = to_b64(frame, max_w=CW)

    full_html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',Arial,sans-serif;}}
body{{background:#f4f6fb;display:flex;gap:12px;padding:8px;min-height:{ch+10}px;}}
#left{{flex:0 0 {cw}px;display:flex;flex-direction:column;gap:6px;}}
canvas{{border:2px solid #1565c0;border-radius:12px;cursor:crosshair;
    display:block;width:{cw}px;height:{ch}px;
    box-shadow:0 3px 12px rgba(21,101,192,.2);}}
#status{{font-size:12px;font-weight:600;color:#1565c0;min-height:18px;padding:2px 0;}}
#zonelist{{font-size:11px;color:#555;line-height:1.8;}}
#right{{flex:1;display:flex;flex-direction:column;gap:10px;}}
.section-title{{font-size:10px;font-weight:700;letter-spacing:.1em;
    text-transform:uppercase;color:#1565c0;margin-bottom:4px;}}
.warn-box{{background:#fff8e1;border-left:4px solid #ffa000;
    border-radius:0 8px 8px 0;padding:8px 10px;
    font-size:12px;color:#e65100;line-height:1.6;}}
.info-box{{background:#e3f2fd;border-left:4px solid #1565c0;
    border-radius:0 8px 8px 0;padding:8px 10px;
    font-size:12px;color:#0d47a1;line-height:1.6;}}
.green-box{{background:#e8f5e9;border-left:4px solid #43a047;
    border-radius:0 8px 8px 0;padding:8px 10px;
    font-size:12px;color:#1b5e20;line-height:1.6;margin-bottom:4px;}}
.btn-row{{display:flex;gap:8px;}}
button{{border:none;border-radius:8px;padding:9px 14px;font-size:13px;
    font-weight:600;cursor:pointer;transition:opacity .15s;font-family:inherit;}}
button:hover{{opacity:.82;}}
button:active{{opacity:.65;}}
#btn-undo{{background:#ef5350;color:#fff;flex:1;}}
#btn-clear{{background:#78909c;color:#fff;flex:1;}}
#btn-save{{background:linear-gradient(135deg,#1565c0,#0d47a1);
    color:#fff;width:100%;font-size:14px;padding:11px;
    box-shadow:0 3px 10px rgba(21,101,192,.35);}}
#jfallback{{display:none;flex-direction:column;gap:6px;}}
#jfallback .label{{font-size:11px;font-weight:700;color:#b71c1c;}}
#jtxt{{width:100%;font-size:10px;font-family:monospace;
    border:1.5px solid #e57373;border-radius:6px;
    padding:6px;resize:none;background:#fff;color:#333;height:70px;}}
#btn-copy{{background:#1565c0;color:#fff;border:none;border-radius:7px;
    padding:7px;font-size:12px;font-weight:600;cursor:pointer;width:100%;}}
</style>
</head><body>
<div id="left">
  <canvas id="C" width="{cw}" height="{ch}"></canvas>
  <div id="status">🖱 Click and drag to draw a parking zone</div>
  <div id="zonelist"></div>
</div>
<div id="right">
  <div class="section-title">✏️ Drawing Controls</div>
  <div class="warn-box">
    <b>Draw tightly around each parking ROW or individual space.</b><br>
    Make sure parked cars will be INSIDE your box.
  </div>
  <div class="btn-row">
    <button id="btn-undo" onclick="undo()">↩ Undo Last</button>
    <button id="btn-clear" onclick="clr()">🗑 Clear All</button>
  </div>
  <button id="btn-save" onclick="save()">✅ Save &amp; Continue →</button>
  <div class="green-box">
    <b>Option A — Draw over a whole ROW</b><br>
    System finds slot lines inside automatically.
  </div>
  <div class="info-box">
    <b>Option B — One small box per slot</b><br>
    Most precise. Draw directly on each space.
  </div>
  <div class="info-box" id="redirect-hint">
    <b>If Save doesn't navigate:</b><br>
    A JSON block will appear below — copy and paste it below the canvas.
  </div>
  <div id="jfallback">
    <div class="label">📋 Copy this JSON and paste it below the canvas:</div>
    <textarea id="jtxt" readonly></textarea>
    <button id="btn-copy" onclick="copyJson()">📋 Copy JSON to Clipboard</button>
  </div>
</div>
<script>
const C = document.getElementById('C');
const ctx = C.getContext('2d');
const img = new Image();
img.src = 'data:image/jpeg;base64,{b64}';
let R = [], drawing = false, sx = 0, sy = 0;
img.onload = () => redraw();
function getXY(e) {{
  const r = C.getBoundingClientRect();
  return [(e.clientX-r.left)*(C.width/r.width),(e.clientY-r.top)*(C.height/r.height)];
}}
function redraw() {{
  ctx.clearRect(0,0,C.width,C.height);
  ctx.drawImage(img,0,0);
  R.forEach((r,i) => {{
    ctx.fillStyle='rgba(0,230,118,.22)';
    ctx.strokeStyle='#00c853';ctx.lineWidth=2.5;
    ctx.fillRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
    ctx.strokeRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
    ctx.fillStyle='rgba(0,0,0,.75)';ctx.fillRect(r.x1,r.y1,24,16);
    ctx.fillStyle='#fff';ctx.font='bold 10px Inter';
    ctx.fillText('Z'+(i+1),r.x1+3,r.y1+12);
  }});
  const s=document.getElementById('status');
  s.textContent=R.length===0?'🖱 Click and drag to draw a zone':
    '✅ '+R.length+' zone'+(R.length!==1?'s':'')+' drawn — draw more or click Save';
  const el=document.getElementById('zonelist');
  el.innerHTML=R.length===0?'':
    '<b style="font-size:10px;color:#777;">Zones: </b>'+
    R.map((r,i)=>`<b>Z${{i+1}}</b> ${{r.x2-r.x1}}×${{r.y2-r.y1}}px`).join(' · ');
}}
C.addEventListener('mousedown',e=>{{[sx,sy]=getXY(e);drawing=true;}});
C.addEventListener('mousemove',e=>{{
  if(!drawing)return;const[cx,cy]=getXY(e);redraw();
  ctx.fillStyle='rgba(0,230,118,.18)';ctx.strokeStyle='#00c853';ctx.lineWidth=2;
  ctx.fillRect(sx,sy,cx-sx,cy-sy);ctx.strokeRect(sx,sy,cx-sx,cy-sy);
}});
C.addEventListener('mouseup',e=>{{
  if(!drawing)return;drawing=false;const[cx,cy]=getXY(e);
  const x1=Math.round(Math.min(sx,cx)),x2=Math.round(Math.max(sx,cx));
  const y1=Math.round(Math.min(sy,cy)),y2=Math.round(Math.max(sy,cy));
  if(Math.abs(x2-x1)>10&&Math.abs(y2-y1)>10)R.push({{x1,y1,x2,y2}});
  redraw();
}});
function undo(){{if(R.length){{R.pop();redraw();}}}}
function clr(){{R=[];redraw();document.getElementById('jfallback').style.display='none';}}
function save(){{
  if(!R.length){{document.getElementById('status').textContent='⚠️ Draw at least one zone first!';return;}}
  const j=JSON.stringify(R);
  try{{const base=window.parent.location.href.split('?')[0];
    window.parent.location.href=base+'?pvs='+encodeURIComponent(j);}}
  catch(err){{showFallback(j);}}
  setTimeout(()=>{{showFallback(j);}},800);
}}
function showFallback(j){{
  const fb=document.getElementById('jfallback');fb.style.display='flex';
  document.getElementById('jtxt').value=j;
  document.getElementById('redirect-hint').style.display='none';
  document.getElementById('status').textContent='📋 Copy the JSON below and paste it below the canvas';
}}
function copyJson(){{
  const t=document.getElementById('jtxt');t.select();t.setSelectionRange(0,99999);
  navigator.clipboard.writeText(t.value)
    .then(()=>{{document.getElementById('btn-copy').textContent='✅ Copied!';
      setTimeout(()=>document.getElementById('btn-copy').textContent='📋 Copy JSON to Clipboard',2000);}})
    .catch(()=>document.execCommand('copy'));
}}
</script>
</body></html>"""

    total_height = ch + 280
    st.components.v1.html(full_html, height=total_height, scrolling=False)

    st.markdown("""
    <div class="ibox" style="margin-top:.5rem;font-size:.78rem;">
    <b>📋 Fallback — paste JSON here if Save didn't redirect:</b>
    </div>""", unsafe_allow_html=True)

    mj = st.text_area("", height=60,
                       placeholder='[{"x1":100,"y1":50,"x2":300,"y2":150}, ...]  ← paste JSON here',
                       key="mj", label_visibility="collapsed")
    if mj and mj.strip():
        stripped = mj.strip()
        if stripped.startswith('[') and '"x1"' in stripped:
            try:
                raw = json.loads(stripped)
                zones = {f"zone_{i}": {
                    "x1": int(float(r["x1"])), "y1": int(float(r["y1"])),
                    "x2": int(float(r["x2"])), "y2": int(float(r["y2"]))}
                    for i, r in enumerate(raw)}
                st.success(f"✅ {len(zones)} zones loaded!")
                if st.button("➡️  Proceed to Calibration", key="proceed_calib"):
                    st.session_state.slots = zones
                    st.session_state.stage = "calibrate"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Invalid JSON: {e}")
        else:
            st.warning("⚠️ That doesn't look like JSON. Copy the text starting with [ from the canvas.")

    col_back, _ = st.columns([1, 3])
    with col_back:
        if st.button("← Back to Upload", key="draw_back"):
            st.session_state.stage = "upload"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "calibrate":
    show_topbar()
    stepbar(3)

    st.markdown("""
    <div class="ibox"><b>🎯 Smart Calibration — v2 Fixed Approach</b><br><br>
    <b>YOLO is now PRIMARY:</b> YOLOv8 detects vehicles in every frame with IoU overlap
    checking (≥8% overlap with slot = OCCUPIED). This is reliable regardless of the
    reference frame quality.<br><br>
    <b>Background subtraction is SECONDARY:</b> Only used as a fallback for vehicles
    YOLO misses (unusual angles, low confidence). Thresholds raised to avoid false
    positives.<br><br>
    <b>Slot detection fix:</b> Minimum slot size enforced (no more thin slivers).
    High-confidence YOLO detections bypass the vote buffer for instant response.
    </div>""", unsafe_allow_html=True)

    zones  = st.session_state.slots
    frame  = st.session_state.first_frame
    scale  = st.session_state.scale

    preview_slots = zones_to_slots(zones, frame, scale)
    n = len(preview_slots)

    st.markdown(f"**{len(zones)} zone(s) drawn** → **{n} individual slots detected** from road line analysis.")

    prev_f  = frame.copy()
    fh_p, fw_p = prev_f.shape[:2]
    palette = [(0,200,60),(0,180,220),(220,120,0),(180,0,200),(0,100,200),(200,160,0)]
    for sid, slot in preview_slots.items():
        x1 = max(0, slot["x1"]); y1 = max(0, slot["y1"])
        x2 = min(fw_p, slot["x2"]); y2 = min(fh_p, slot["y2"])
        ci = int(sid.split("_")[1]) % len(palette)
        cv2.rectangle(prev_f, (x1, y1), (x2, y2), palette[ci], 2)
        sn = int(sid.split("_")[1]) + 1
        (tw, th), _ = cv2.getTextSize(f"P{sn}", cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
        cv2.rectangle(prev_f, (x1+1, y1+1), (x1+tw+5, y1+th+4), palette[ci], -1)
        cv2.putText(prev_f, f"P{sn}", (x1+3, y1+th+1),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1, cv2.LINE_AA)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.image(cv2.cvtColor(small(prev_f, 500), cv2.COLOR_BGR2RGB),
                 caption=f"Preview: {n} slots detected from {len(zones)} zone(s)",
                 use_column_width=False)
    with col2:
        st.markdown(f"""
        <div class="sbox"><b>✅ {n} parking slots detected</b><br>
        From {len(zones)} drawn zone(s).<br>
        Detection uses IoU overlap — cars don't need to be perfectly centered.
        </div>
        <div class="wbox"><b>Slot count wrong?</b><br>
        Go back and redraw zones. Draw tight boxes around each parking row.
        For individual spaces, draw one box per space.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔧  Run Calibration & Start Detection"):
            prog = st.progress(0, text="Reading frames for smart reference...")
            cap  = cv2.VideoCapture(st.session_state.video_path)

            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 120
            step = max(1, total_vid_frames // 120)
            frames_gray = []
            fi = 0
            while len(frames_gray) < 120:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, f = cap.read()
                if not ret:
                    break
                frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
                fi += step
                prog.progress(min(len(frames_gray)/120, 0.9),
                              text=f"Sampling frame {len(frames_gray)}/120...")
            cap.release()

            if frames_gray:
                ref_max, ref_min = build_smart_reference(frames_gray)
                prog.progress(1.0, text="✅ Smart reference built!")
                time.sleep(0.4)
                st.session_state.ref_frame     = ref_max
                st.session_state.ref_max       = ref_max
                st.session_state.ref_min       = ref_min
                st.session_state.slots         = preview_slots
                st.session_state.calib_done    = True
                st.session_state.events        = []
                st.session_state.occ_history   = []
                st.session_state.dwell_times   = []
                st.session_state.slot_entry_times = {}
                st.session_state.max_cars_seen = 0  # FIX: reset so new session starts clean
                st.session_state.stage         = "detect"
                st.rerun()
            else:
                st.error("❌ Could not read frames from video.")

        if st.button("← Redraw Zones"):
            st.session_state.stage = "draw"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "detect":
    show_topbar(live=True)
    stepbar(4)

    if st.session_state.show_summary:
        s     = st.session_state.summary
        total = s["total"]; free = s["free"]; occ = s["occ"]
        cars  = s["cars"];  peak = s["peak"]
        pct   = int(occ / total * 100) if total else 0
        free_pct = 100 - pct
        bar_color = "#43a047" if free_pct > 60 else "#ffa000" if free_pct > 30 else "#e53935"
        msg = ("🎉 Plenty of parking available!" if free_pct > 60 else
               "⚠️ Parking is getting full!"    if free_pct > 30 else
               "🚨 Parking lot is almost full!")
        msg_bg = ("#e8f5e9" if free_pct > 60 else
                  "#fff8e1" if free_pct > 30 else "#ffebee")
        msg_c  = ("#1b5e20" if free_pct > 60 else
                  "#e65100" if free_pct > 30 else "#b71c1c")
        dwell  = st.session_state.dwell_times
        avg_d  = f"{np.mean(dwell):.1f}s" if dwell else "N/A"

        st.markdown(f"""
        <div class="sum-wrap">
          <div class="sum-header">
            <div class="sum-icon">🚗</div>
            <div class="sum-title">Parking Session Summary</div>
            <div class="sum-sub">Session: {st.session_state.session_start or 'N/A'}</div>
          </div>
          <div class="sum-grid">
            <div class="sm sm-t"><div class="n">{total}</div><div class="l">Total Slots</div></div>
            <div class="sm sm-v"><div class="n">{free}</div><div class="l">🟢 Vacant</div></div>
            <div class="sm sm-o"><div class="n">{occ}</div><div class="l">🔴 Occupied</div></div>
            <div class="sm sm-c"><div class="n">{cars}</div><div class="l">🚗 Cars Seen</div></div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-bottom:1rem;">
            <div class="sm sm-t"><div class="n" style="font-size:1.6rem;">{peak}</div>
              <div class="l">Peak Occupied</div></div>
            <div class="sm sm-v"><div class="n" style="font-size:1.6rem;">{avg_d}</div>
              <div class="l">Avg Dwell</div></div>
            <div class="sm sm-c"><div class="n" style="font-size:1.6rem;">{len(st.session_state.events)}</div>
              <div class="l">Events</div></div>
          </div>
          <div class="sum-bar-wrap">
            <div class="sum-bar" style="width:{pct}%;background:{bar_color};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
              font-size:.78rem;color:#9e9e9e;font-weight:600;margin-bottom:.8rem;">
            <span>🟢 {free} free ({free_pct}%)</span>
            <span>🔴 {occ} occupied ({pct}%)</span>
          </div>
          <div class="sum-status" style="background:{msg_bg};color:{msg_c};">{msg}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pdf_data, mime, fname = make_report(
            s, st.session_state.events,
            st.session_state.dwell_times,
            st.session_state.session_start)
        ext = fname.split('.')[-1].upper()
        st.download_button(f"📥 Download Session Report ({ext})",
                           data=pdf_data, file_name=fname, mime=mime)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Detect Again"):
                st.session_state.show_summary = False
                st.session_state.events       = []
                st.session_state.occ_history  = []
                st.session_state.dwell_times  = []
                st.session_state.slot_entry_times = {}
                st.rerun()
        with c2:
            if st.button("🖊 Redraw Slots"):
                st.session_state.show_summary = False
                st.session_state.stage = "draw"; st.rerun()
        st.stop()

    # ── Live detection ────────────────────────────────────────────────────────
    model  = load_model()
    slots  = st.session_state.slots
    total  = len(slots)
    ref_max = st.session_state.get("ref_max", st.session_state.ref_frame)
    ref_min = st.session_state.get("ref_min", st.session_state.ref_frame)

    c_vid, c_stat = st.columns([1.1, 1])

    with c_vid:
        st.markdown('<div class="card"><div class="card-title">📹 Live Feed</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;gap:1rem;align-items:center;font-size:.78rem;
            color:#424242;background:#f5f7ff;border-radius:8px;
            padding:.45rem .9rem;margin-bottom:.7rem;border:1px solid #e3eaf7;flex-wrap:wrap;">
          <span>🟢 <b>FREE</b> = vacant (glowing green border + dot)</span>
          <span>🔴 <b>FULL</b> = occupied (red border)</span>
          <span style="color:#888;">🔵 outline = YOLO car</span>
        </div>""", unsafe_allow_html=True)
        vid_ph   = st.empty()
        alert_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with c_stat:
        st.markdown('<div class="card"><div class="card-title">📊 Live Statistics</div>',
                    unsafe_allow_html=True)
        stat_ph = st.empty()
        prog_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">📈 Analytics</div>',
                    unsafe_allow_html=True)
        an_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">📋 Real-time Events</div>',
                    unsafe_allow_html=True)
        ev_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">⚙️ Controls</div>',
                    unsafe_allow_html=True)
        stop_btn = st.button("⏹  Stop & Show Summary")
        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
        if st.button("← Redraw Slots"):
            st.session_state.stage = "draw"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Detection loop ─────────────────────────────────────────────────────────
    cap  = cv2.VideoCapture(st.session_state.video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    fc   = 0; boxes = []
    prev = {sid: False for sid in slots}
    blink_phase = 0; blink_counter = 0
    alert_state = None

    # FIX — Cars Seen bug: persist max_cars in session state so it survives the
    # Streamlit rerun triggered by stop_btn. A plain local variable gets lost.
    if "max_cars_seen" not in st.session_state:
        st.session_state.max_cars_seen = 0

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        fc += 1; fh, fw = frame.shape[:2]

        # FIX — Run YOLO on every frame (was fc%2==0, so frame 1 was always skipped
        # and if stop was pressed immediately, YOLO had never run → boxes=[] → cars=0).
        # Now runs every frame; throttle only if performance is poor.
        boxes = detect_vehicles_yolo(frame, model)
        # Update session-state max immediately so it's never lost on rerun
        st.session_state.max_cars_seen = max(
            st.session_state.max_cars_seen, len(boxes))

        # Blink timer
        blink_counter += 1
        if blink_counter >= max(1, int(fps * 0.7)):
            blink_phase   = 1 - blink_phase
            blink_counter = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for sid, slot in slots.items():
            # FIX BUG 1 + 6: check_slot_yolo uses IoU, in video pixel coordinates
            yolo_occ, yolo_ovlp  = check_slot_yolo(slot, boxes, fw, fh)
            bg_score             = check_slot_bg(gray, ref_max, ref_min, slot)
            now                  = decide_occupied(yolo_occ, yolo_ovlp, bg_score, slot)
            slot["occupied"]     = now
            slot["score"]        = bg_score

            if now != prev[sid]:
                n  = int(sid.split("_")[1]) + 1
                ts = datetime.now().strftime("%H:%M:%S")
                if now:
                    st.session_state.events.append(
                        f"🔴 {ts} — P{n} OCCUPIED (car entered)")
                    st.session_state.slot_entry_times[sid] = datetime.now()
                else:
                    st.session_state.events.append(
                        f"🟢 {ts} — P{n} VACANT (car left)")
                    if sid in st.session_state.slot_entry_times:
                        dwell = (datetime.now() -
                                 st.session_state.slot_entry_times[sid]).total_seconds()
                        st.session_state.dwell_times.append(dwell)
                        del st.session_state.slot_entry_times[sid]
            prev[sid] = now

        ann = draw_result(frame.copy(), slots, boxes, blink_phase=blink_phase)
        vid_ph.image(cv2.cvtColor(small(ann, 460), cv2.COLOR_BGR2RGB),
                     use_column_width=False)

        occ  = sum(1 for s in slots.values() if s["occupied"])
        free = total - occ
        pct  = int(occ / total * 100) if total else 0
        cars = len(boxes)
        # max_cars_seen already updated above right after YOLO runs
        st.session_state.occ_history.append(occ)
        bar_c = "#e53935" if pct > 70 else "#ffa000" if pct > 40 else "#43a047"

        stat_ph.markdown(f"""
        <div class="srow">
          <div class="sc sc-t"><div class="sc-n">{total}</div><div class="sc-l">Total</div></div>
          <div class="sc sc-v"><div class="sc-n">{free}</div><div class="sc-l">🟢 Vacant</div></div>
          <div class="sc sc-o"><div class="sc-n">{occ}</div><div class="sc-l">🔴 Occupied</div></div>
          <div class="sc sc-c"><div class="sc-n">{cars}</div><div class="sc-l">🚗 Cars</div></div>
        </div>""", unsafe_allow_html=True)

        prog_ph.markdown(f"""
        <div class="pw">
          <div class="pf" style="width:{pct}%;background:{bar_c};"></div>
        </div>
        <div class="pl">
          <span>🟢 {free} free ({100-pct}%)</span>
          <span>🔴 {occ} occupied ({pct}%)</span>
        </div>""", unsafe_allow_html=True)

        dwell_list = st.session_state.dwell_times
        avg_d = f"{np.mean(dwell_list):.0f}s" if dwell_list else "—"
        peak  = max(st.session_state.occ_history) if st.session_state.occ_history else 0
        an_ph.markdown(f"""
        <div class="an-row">
          <div class="an-chip"><div class="n">{peak}</div><div class="l">Peak Occ.</div></div>
          <div class="an-chip"><div class="n">{avg_d}</div><div class="l">Avg Dwell</div></div>
          <div class="an-chip"><div class="n">{len(st.session_state.events)}</div>
              <div class="l">Events</div></div>
        </div>""", unsafe_allow_html=True)

        # Alert banner
        if pct >= 95:
            if alert_state != "full":
                alert_state = "full"
            alert_ph.markdown(
                '<div class="alert-full">🚨 PARKING LOT FULL — Redirect incoming vehicles!</div>',
                unsafe_allow_html=True)
        elif pct < 70 and alert_state == "full":
            alert_state = "ok"
            alert_ph.markdown(
                '<div class="alert-ok">✅ Spaces now available!</div>',
                unsafe_allow_html=True)

        if st.session_state.events:
            evs = "".join(f"<div>{e}</div>" for e in st.session_state.events[-5:])
            ev_ph.markdown(f'<div class="ev-box">{evs}</div>', unsafe_allow_html=True)

        time.sleep(1 / fps)

    cap.release()

    occ_f  = sum(1 for s in slots.values() if s["occupied"])
    peak_f = max(st.session_state.occ_history) if st.session_state.occ_history else 0
    st.session_state.summary = {
        "total": total, "free": total - occ_f,
        "occ":   occ_f, "cars": st.session_state.max_cars_seen,  # FIX: use persisted value
        "peak":  peak_f,
    }
    st.session_state.show_summary = True
    st.rerun()