"""
ParkVision Pro — Final Version
==============================
SLOT DETECTION:
  - Draw rectangle over a PARKING ROW
  - Sobel edge detection finds parking LINE MARKINGS inside
  - Each gap between lines = one parking slot
  - Cars in slots are counted → TOTAL = vacant + occupied
  
UI: Stats RIGHT of video (not below)
PDF: HTML fallback always works
"""

import streamlit as st
import cv2
import json
import numpy as np
import tempfile
import time
import base64
import io
from datetime import datetime

st.set_page_config(page_title="ParkVision Pro", layout="wide",
                   initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# CSS — original UI (stats right of video, white background, card layout)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main,.main .block-container{
    background:#f4f6fb !important;font-family:'Inter',sans-serif !important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;}
.block-container{padding:1.4rem 2rem !important;max-width:1350px;}

.topbar{background:linear-gradient(135deg,#0a2463,#1565c0,#1976d2);
    border-radius:18px;padding:1.3rem 1.8rem;display:flex;align-items:center;
    gap:1rem;margin-bottom:1.4rem;box-shadow:0 6px 24px rgba(10,36,99,.35);}
.topbar-icon{font-size:2.4rem;}
.topbar-title{font-size:1.65rem;font-weight:800;color:#fff;margin:0;}
.topbar-sub{font-size:.79rem;color:#90caf9;margin:.2rem 0 0;}
.live-pill{margin-left:auto;background:rgba(255,60,60,.9);border-radius:999px;
    padding:.28rem 1rem;font-size:.72rem;color:#fff;font-weight:700;
    animation:blink 1.4s infinite;white-space:nowrap;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.5;}}

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

/* Stat cards - 4 across */
.srow{display:flex;gap:.7rem;margin-bottom:.9rem;}
.sc{flex:1;border-radius:14px;padding:1rem .6rem;text-align:center;}
.sc-t{background:linear-gradient(135deg,#e8eaf6,#c5cae9);border:1.5px solid #9fa8da;}
.sc-v{background:linear-gradient(135deg,#e8f5e9,#a5d6a7);border:1.5px solid #81c784;}
.sc-o{background:linear-gradient(135deg,#ffebee,#ef9a9a);border:1.5px solid #e57373;}
.sc-c{background:linear-gradient(135deg,#fff8e1,#ffe082);border:1.5px solid #ffd54f;}
.sc-n{font-size:2.6rem;font-weight:800;line-height:1;}
.sc-t .sc-n{color:#283593;}.sc-v .sc-n{color:#1b5e20;}
.sc-o .sc-n{color:#b71c1c;}.sc-c .sc-n{color:#e65100;}
.sc-l{font-size:.63rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem;}
.sc-t .sc-l{color:#5c6bc0;}.sc-v .sc-l{color:#388e3c;}
.sc-o .sc-l{color:#e53935;}.sc-c .sc-l{color:#f57f17;}

.pw{background:#e0e0e0;border-radius:999px;height:12px;overflow:hidden;margin:.7rem 0 .3rem;}
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
    padding:.75rem 1rem;font-size:.76rem;color:#424242;line-height:2.1;
    max-height:140px;overflow-y:auto;}
.an-row{display:flex;gap:.6rem;margin:.4rem 0;}
.an-chip{flex:1;background:#f8f9ff;border:1px solid #e3eaf7;border-radius:10px;
    padding:.55rem .4rem;text-align:center;}
.an-chip .n{font-size:1.25rem;font-weight:800;color:#1565c0;}
.an-chip .l{font-size:.58rem;color:#9e9e9e;text-transform:uppercase;
    letter-spacing:.07em;font-weight:600;margin-top:.1rem;}

.alert-full{background:#ffebee;border:2px solid #e53935;border-radius:10px;
    padding:.65rem 1rem;font-size:.82rem;color:#b71c1c;font-weight:700;
    text-align:center;animation:alertp 1.6s infinite;}
@keyframes alertp{0%,100%{box-shadow:0 0 0 0 rgba(229,57,53,.3);}
    50%{box-shadow:0 0 0 8px rgba(229,57,53,0);}}
.alert-ok{background:#e8f5e9;border:2px solid #43a047;border-radius:10px;
    padding:.65rem 1rem;font-size:.82rem;color:#1b5e20;font-weight:700;text-align:center;}

/* Splash */
.splash{background:linear-gradient(135deg,#0a2463,#1565c0,#1976d2);
    border-radius:22px;padding:3rem 2rem;text-align:center;
    box-shadow:0 12px 40px rgba(10,36,99,.4);margin:1rem 0;}
.splash-icon{font-size:5rem;margin-bottom:.8rem;}
.splash-title{font-size:2.4rem;font-weight:800;color:#fff;margin-bottom:.4rem;}
.splash-sub{font-size:.98rem;color:#90caf9;margin-bottom:1.8rem;line-height:1.7;}
.feat-row{display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.8rem;}
.feat{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);
    border-radius:999px;padding:.38rem 1rem;font-size:.77rem;color:#fff;font-weight:600;}

/* Summary */
.sum-wrap{background:#fff;border:2px solid #c5cae9;border-radius:20px;
    padding:2rem;box-shadow:0 4px 20px rgba(0,0,0,.08);}
.sum-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem;}
.sm{border-radius:14px;padding:1rem .7rem;text-align:center;}
.sm-t{background:#e8eaf6;}.sm-v{background:#e8f5e9;}
.sm-o{background:#ffebee;}.sm-c{background:#fff8e1;}
.sm .n{font-size:2.4rem;font-weight:800;line-height:1;}
.sm-t .n{color:#283593;}.sm-v .n{color:#1b5e20;}
.sm-o .n{color:#b71c1c;}.sm-c .n{color:#e65100;}
.sm .l{font-size:.63rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;margin-top:.2rem;}
.sm-t .l{color:#5c6bc0;}.sm-v .l{color:#388e3c;}
.sm-o .l{color:#e53935;}.sm-c .l{color:#f57f17;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in dict(slots={}, video_path=None, first_frame=None,
                ref_gray=None, stage="splash", scale=1.0,
                show_summary=False, summary={},
                events=[], dwell_times=[], slot_entry_times={},
                occ_history=[], max_cars=0,
                session_start=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# URL param receiver
qp = st.query_params
if "pvs" in qp and st.session_state.stage == "draw":
    try:
        raw = json.loads(qp["pvs"])
        zones = {f"zone_{i}":{"x1":int(float(r["x1"])),"y1":int(float(r["y1"])),
                               "x2":int(float(r["x2"])),"y2":int(float(r["y2"]))}
                 for i,r in enumerate(raw)}
        if zones:
            st.session_state.slots  = zones
            st.session_state.stage  = "calibrate"
            st.query_params.clear()
            st.rerun()
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def stepbar(cur):
    labels=["Upload","Draw Zones","Calibrate","Detect"]
    h='<div class="stepbar">'
    for i,l in enumerate(labels,1):
        cls="done" if i<cur else("active" if i==cur else "stp")
        icon="✓" if i<cur else str(i)
        h+=f'<div class="stp {cls}"><div class="stp-n">{icon}</div>{l}</div>'
        if i<len(labels): h+='<div class="stp-line"></div>'
    h+='</div>'
    st.markdown(h,unsafe_allow_html=True)

def topbar(live=False):
    pill='<div class="live-pill">● LIVE</div>' if live else ''
    st.markdown(f"""<div class="topbar">
      <div class="topbar-icon">🅿️</div>
      <div><p class="topbar-title">ParkVision Pro</p>
      <p class="topbar-sub">Real-time parking detector · Line Detection + Background Model + YOLOv8</p></div>
      {pill}</div>""",unsafe_allow_html=True)

@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8s.pt")

def small(f,w=460):
    h,fw=f.shape[:2]
    if fw>w: f=cv2.resize(f,(w,int(h*w/fw)))
    return f

def to_b64(f,mw=700):
    h,w=f.shape[:2]
    if w>mw: f=cv2.resize(f,(mw,int(h*mw/w)))
    _,buf=cv2.imencode(".jpg",f,[cv2.IMWRITE_JPEG_QUALITY,88])
    return base64.b64encode(buf).decode(),f.shape[1],f.shape[0]

# ══════════════════════════════════════════════════════════════════════════════
# SLOT LINE DETECTION
# Detects parking bay dividers inside a drawn zone using Sobel edges.
# Each gap between detected lines = one parking slot.
# Total slots = vacant + occupied (counted from video lines, not just cars).
# ══════════════════════════════════════════════════════════════════════════════
def detect_slots_in_zone(frame, zone, scale):
    """
    Detect individual parking slots inside a drawn zone by finding
    parking line markings using edge detection.
    
    Returns list of slot dicts in VIDEO pixel coordinates.
    """
    sx = 1.0 / scale

    # Zone in video pixels
    zx1 = max(0, int(zone["x1"] * sx))
    zy1 = max(0, int(zone["y1"] * sx))
    zx2 = min(frame.shape[1], int(zone["x2"] * sx))
    zy2 = min(frame.shape[0], int(zone["y2"] * sx))
    zw  = zx2 - zx1
    zh  = zy2 - zy1

    if zw < 10 or zh < 10:
        return [make_slot(zx1,zy1,zx2,zy2)]

    roi  = frame[zy1:zy2, zx1:zx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ── Detect vertical lines (columns of parking spaces side by side) ─────
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_proj = np.abs(sobelx).mean(axis=0)
    col_smooth = np.convolve(col_proj,
                             np.ones(max(3,zw//30))/max(3,zw//30),
                             mode='same')
    v_peaks = find_peaks(col_smooth, zw, min_dist_frac=0.06)

    # ── Detect horizontal lines (rows stacked vertically) ─────────────────
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_proj = np.abs(sobely).mean(axis=1)
    row_smooth = np.convolve(row_proj,
                             np.ones(max(3,zh//20))/max(3,zh//20),
                             mode='same')
    h_peaks = find_peaks(row_smooth, zh, min_dist_frac=0.08)

    # ── Choose orientation with more lines ─────────────────────────────────
    slots = []
    if len(v_peaks) >= len(h_peaks) and len(v_peaks) > 0:
        # Vertical dividers → horizontal layout
        xs = sorted([0] + v_peaks + [zw])
        for i in range(len(xs)-1):
            if xs[i+1]-xs[i] < 8: continue
            slots.append(make_slot(zx1+xs[i], zy1, zx1+xs[i+1], zy2))
    elif len(h_peaks) > 0:
        # Horizontal dividers → vertical layout
        ys = sorted([0] + h_peaks + [zh])
        for i in range(len(ys)-1):
            if ys[i+1]-ys[i] < 8: continue
            slots.append(make_slot(zx1, zy1+ys[i], zx2, zy1+ys[i+1]))
    else:
        # ── Fallback: aspect ratio heuristic ─────────────────────────────
        aspect = zw / max(zh, 1)
        if aspect >= 1.8:
            # Wide → multiple side-by-side slots
            n = max(1, round(aspect / 1.6))
            sw = zw // n
            for i in range(n):
                x1s = zx1 + i*sw
                x2s = zx1 + (i+1)*sw if i<n-1 else zx2
                slots.append(make_slot(x1s, zy1, x2s, zy2))
        elif aspect <= 0.55:
            # Tall → stacked slots
            n = max(1, round((1/aspect) / 1.6))
            sh = zh // n
            for i in range(n):
                y1s = zy1 + i*sh
                y2s = zy1 + (i+1)*sh if i<n-1 else zy2
                slots.append(make_slot(zx1, y1s, zx2, y2s))
        else:
            slots.append(make_slot(zx1, zy1, zx2, zy2))

    return slots if slots else [make_slot(zx1,zy1,zx2,zy2)]


def find_peaks(signal, dim, min_dist_frac=0.06):
    """Find peaks in a 1D signal above mean+0.7*std, with minimum distance."""
    if len(signal) == 0:
        return []
    mean = signal.mean()
    std  = signal.std()
    thr  = mean + 0.7 * std
    min_d = max(10, int(dim * min_dist_frac))
    peaks = []
    for i in range(min_d, len(signal)-min_d):
        if signal[i] > thr:
            window = signal[max(0,i-min_d):i+min_d]
            if signal[i] == window.max():
                peaks.append(i)
    return peaks


def make_slot(x1,y1,x2,y2):
    return {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"occupied":False,"ema":0.0}


def zones_to_slots(zones, frame, scale):
    all_slots = {}
    idx = 0
    for zid, zone in zones.items():
        for s in detect_slots_in_zone(frame, zone, scale):
            all_slots[f"slot_{idx}"] = s
            idx += 1
    return all_slots

# ══════════════════════════════════════════════════════════════════════════════
# OCCUPANCY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def slot_bg_check(frame_gray, ref_gray, slot, thresh=38, min_frac=0.20):
    x1=max(0,slot["x1"]); y1=max(0,slot["y1"])
    x2=min(frame_gray.shape[1],slot["x2"])
    y2=min(frame_gray.shape[0],slot["y2"])
    if x2<=x1 or y2<=y1: return False, 0.0
    cur = frame_gray[y1:y2,x1:x2].astype(np.float32)
    ref = ref_gray[y1:y2,x1:x2].astype(np.float32)
    diff = np.abs(cur-ref)
    frac = (diff>thresh).sum() / max(1,diff.size)
    return bool(frac>min_frac), float(frac)

def slot_yolo_check(slot, boxes, fw, fh):
    x1=max(0,slot["x1"]); y1=max(0,slot["y1"])
    x2=min(fw,slot["x2"]); y2=min(fh,slot["y2"])
    sa=max(1,(x2-x1)*(y2-y1))
    for box in boxes:
        bx1,by1,bx2,by2=[float(v) for v in box[:4]]
        cx=(bx1+bx2)/2; cy=(by1+by2)/2
        if x1<cx<x2 and y1<cy<y2: return True
        ix1,iy1=max(bx1,x1),max(by1,y1)
        ix2,iy2=min(bx2,x2),min(by2,y2)
        if max(0,ix2-ix1)*max(0,iy2-iy1)/sa > 0.15: return True
    return False

def update_slot_state(slot, frame_gray, ref_gray, boxes, fw, fh, thresh):
    bg_occ, bg_frac = slot_bg_check(frame_gray, ref_gray, slot, thresh)
    yolo_occ        = slot_yolo_check(slot, boxes, fw, fh)

    # Combined signal
    if bg_frac > 0.55:      raw = 0.92
    elif bg_occ and yolo_occ: raw = 0.82
    elif bg_occ:              raw = bg_frac * 0.65
    elif yolo_occ:            raw = 0.45
    else:                     raw = bg_frac * 0.25

    ema = 0.25*raw + 0.75*slot.get("ema", 0.0)
    slot["ema"] = float(ema)

    # Hysteresis thresholds
    if ema > 0.30:   occ = True
    elif ema < 0.15: occ = False
    else:            occ = slot.get("occupied", False)

    slot["occupied"] = occ
    return occ

def draw_detection(frame, slots, boxes):
    fh,fw = frame.shape[:2]
    # YOLO car outlines
    for box in boxes:
        bx1,by1,bx2,by2=[int(v) for v in box[:4]]
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,200,255),2)
        cv2.putText(frame,"Car",(bx1,max(by1-5,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.38,(0,200,255),1,cv2.LINE_AA)
    # Parking slots
    for sid,slot in slots.items():
        x1=max(0,slot["x1"]); y1=max(0,slot["y1"])
        x2=min(fw,slot["x2"]); y2=min(fh,slot["y2"])
        occ=slot.get("occupied",False)
        ov=frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,200) if occ else (0,170,50),-1)
        cv2.addWeighted(ov,.22,frame,.78,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,155) if occ else (0,130,30),2)
        label="FULL" if occ else "FREE"
        bg=(0,0,135) if occ else (0,110,25)
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.38,1)
        cv2.rectangle(frame,(x1+2,y1+2),(x1+tw+6,y1+th+5),bg,-1)
        cv2.putText(frame,label,(x1+4,y1+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        n=int(sid.split("_")[1])+1
        cv2.putText(frame,f"P{n}",(x1+3,y2-5),
                    cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,200),1,cv2.LINE_AA)
    return frame

# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION (PDF if reportlab available, else HTML)
# ══════════════════════════════════════════════════════════════════════════════
def make_report(summary, events, dwell_times, session_start):
    total=summary.get("total",0); free=summary.get("free",0)
    occ=summary.get("occ",0); cars=summary.get("cars",0)
    pct=int(occ/total*100) if total else 0
    avg_d=f"{np.mean(dwell_times):.1f}s" if dwell_times else "N/A"
    peak=summary.get("peak",0); now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle)
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER

        buf=io.BytesIO()
        doc=SimpleDocTemplate(buf,pagesize=A4,
            rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
        styles=getSampleStyleSheet()
        story=[]
        T=ParagraphStyle('T',fontSize=22,textColor=colors.HexColor('#0a2463'),
                          spaceAfter=4,alignment=TA_CENTER,fontName='Helvetica-Bold')
        S=ParagraphStyle('S',fontSize=10,textColor=colors.HexColor('#78909c'),
                          alignment=TA_CENTER,spaceAfter=12)
        H=ParagraphStyle('H',fontSize=13,textColor=colors.HexColor('#1565c0'),
                          spaceBefore=14,spaceAfter=6,fontName='Helvetica-Bold')
        story.append(Paragraph("ParkVision Pro",T))
        story.append(Paragraph("Parking Session Report",S))
        story.append(Paragraph(f"Generated: {now} | Session: {session_start or 'N/A'}",S))
        story.append(Spacer(1,12))
        story.append(Paragraph("Session Summary",H))
        data=[["Metric","Value"],
              ["Total Slots",str(total)],["Vacant",str(free)],
              ["Occupied",str(occ)],["Occupancy Rate",f"{pct}%"],
              ["Peak Occupied",str(peak)],["Vehicles Seen",str(cars)],
              ["Avg Dwell Time",avg_d],["Total Events",str(len(events))]]
        t=Table(data,colWidths=[10*cm,7*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTSIZE',(0,0),(-1,-1),10),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#f4f6fb'),colors.white]),
            ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#e3eaf7')),
            ('TOPPADDING',(0,0),(-1,-1),6),
            ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ]))
        story.append(t); story.append(Spacer(1,14))
        if events:
            story.append(Paragraph("Event Log",H))
            ev_data=[["#","Event"]]+[[str(i+1),e] for i,e in enumerate(events)]
            t2=Table(ev_data,colWidths=[1.5*cm,15.5*cm])
            t2.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eaf7')),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,-1),9),
                ('ALIGN',(0,0),(0,-1),'CENTER'),
                ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#e3eaf7')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),
                 [colors.HexColor('#f9faff'),colors.white]),
                ('TOPPADDING',(0,0),(-1,-1),4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4),
            ]))
            story.append(t2)
        doc.build(story); buf.seek(0)
        return buf.read(),"application/pdf","parkvision_report.pdf"

    except ImportError:
        # HTML fallback — always works
        rows="".join(f"<tr><td>{i+1}</td><td>{e}</td></tr>" for i,e in enumerate(events))
        html=f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
        <title>ParkVision Report</title>
        <style>
          body{{font-family:Arial,sans-serif;margin:40px;color:#333;background:#f4f6fb;}}
          h1{{color:#0a2463;}} h2{{color:#1565c0;margin-top:24px;}}
          .grid{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0;}}
          .stat{{background:#fff;border:1.5px solid #e3eaf7;border-radius:12px;
              padding:14px 20px;text-align:center;min-width:110px;}}
          .stat .n{{font-size:2rem;font-weight:800;color:#283593;}}
          .stat .l{{font-size:.75rem;color:#5c6bc0;text-transform:uppercase;}}
          table{{border-collapse:collapse;width:100%;margin-top:8px;background:#fff;
              border-radius:10px;overflow:hidden;}}
          th{{background:#1565c0;color:#fff;padding:9px 12px;text-align:left;}}
          td{{padding:7px 12px;border-bottom:1px solid #e3eaf7;font-size:.88rem;}}
          tr:nth-child(even){{background:#f4f6fb;}}
        </style></head><body>
        <h1>🅿️ ParkVision Pro — Session Report</h1>
        <p style="color:#78909c;">Generated: {now} | Session: {session_start or 'N/A'}</p>
        <div class="grid">
          <div class="stat"><div class="n">{total}</div><div class="l">Total Slots</div></div>
          <div class="stat"><div class="n" style="color:#1b5e20;">{free}</div><div class="l">Vacant</div></div>
          <div class="stat"><div class="n" style="color:#b71c1c;">{occ}</div><div class="l">Occupied</div></div>
          <div class="stat"><div class="n">{pct}%</div><div class="l">Occupancy</div></div>
          <div class="stat"><div class="n">{peak}</div><div class="l">Peak Occ.</div></div>
          <div class="stat"><div class="n">{cars}</div><div class="l">Cars Seen</div></div>
          <div class="stat"><div class="n">{avg_d}</div><div class="l">Avg Dwell</div></div>
        </div>
        <h2>Event Log ({len(events)} events)</h2>
        <table><tr><th>#</th><th>Event</th></tr>{rows}</table>
        </body></html>"""
        return html.encode(),"text/html","parkvision_report.html"

# ══════════════════════════════════════════════════════════════════════════════
# SPLASH
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "splash":
    st.markdown("""
    <div class="splash">
      <div class="splash-icon">🅿️</div>
      <div class="splash-title">ParkVision Pro</div>
      <div class="splash-sub">
        Intelligent real-time parking space monitoring<br>
        Line Detection + Background Subtraction + YOLOv8
      </div>
      <div class="feat-row">
        <span class="feat">🎯 Auto Slot Count from Lines</span>
        <span class="feat">📊 Live Analytics</span>
        <span class="feat">🚨 Lot-Full Alert</span>
        <span class="feat">📋 PDF / HTML Report</span>
        <span class="feat">⏱ Dwell Time Tracking</span>
        <span class="feat">🔄 Real-time Events</span>
      </div>
    </div>""",unsafe_allow_html=True)
    _,cc,_=st.columns([1,1,1])
    with cc:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🚀  Start Detection"):
            st.session_state.stage="upload"
            st.session_state.session_start=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
      <div class="card"><div class="card-title">How It Works</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          1️⃣ Upload parking lot video<br>
          2️⃣ Draw over each parking ROW<br>
          3️⃣ System detects slots from road lines<br>
          4️⃣ Calibrates per-slot background<br>
          5️⃣ Real-time vacant/occupied tracking
        </div></div>
      <div class="card"><div class="card-title">Detection Engine</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          📐 Sobel edge → parking line detection<br>
          🔵 Background subtraction (primary)<br>
          🟡 YOLOv8 vehicle confirmation<br>
          🟢 EMA smoothing (no flickering)<br>
          🔴 Hysteresis (stable transitions)
        </div></div>
      <div class="card"><div class="card-title">What You Get</div>
        <div style="font-size:.82rem;color:#424242;line-height:1.9;">
          📊 Total / Vacant / Occupied live<br>
          🚗 Vehicle count (YOLO)<br>
          📈 Peak occupancy + avg dwell<br>
          🔔 Lot-full alert at 95%<br>
          📥 Download PDF or HTML report
        </div></div>
    </div>""",unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "upload":
    topbar()
    stepbar(1)
    c1,c2=st.columns([1.5,1])
    with c1:
        st.markdown('<div class="card"><div class="card-title">Upload Parking Lot Video</div>',
                    unsafe_allow_html=True)
        up=st.file_uploader("",type=["mp4","avi","mov","mkv"],label_visibility="collapsed")
        if up:
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
            tmp.write(up.read()); tmp.flush()
            cap=cv2.VideoCapture(tmp.name)
            ret,frame=cap.read(); cap.release()
            if ret and frame is not None:
                st.session_state.video_path=tmp.name
                st.session_state.first_frame=frame.copy()
                st.image(cv2.cvtColor(small(frame,480),cv2.COLOR_BGR2RGB),
                         caption="✅ Video loaded",use_column_width=False)
                st.markdown("<br>",unsafe_allow_html=True)
                if st.button("➡️  Next: Draw Parking Zones"):
                    st.session_state.stage="draw"; st.rerun()
            else:
                st.error("❌ Cannot read video. Run: python3 convert_video.py")
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="ibox"><b>📋 How it works</b><br><br>
        Upload your parking lot video — the first frame will be used
        to draw parking zones and calibrate the detection model.<br><br>
        ✅ Works with overhead CCTV cameras<br>
        ✅ Works with angled cameras<br>
        ✅ Cars entering → vacant decreases<br>
        ✅ Cars leaving → vacant increases
        </div>
        <div class="wbox"><b>Video not loading?</b><br>
        <code>python3 convert_video.py</code><br>
        Then upload <b>output.mp4</b>
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DRAW ZONES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "draw":
    topbar()
    stepbar(2)

    frame=st.session_state.first_frame
    CW=700; h0,w0=frame.shape[:2]
    sc=CW/w0; ch=int(h0*sc)
    st.session_state.scale=sc
    b64,cw,_=to_b64(frame,mw=CW)

    st.markdown("""
    <div class="ibox"><b>✏️ Draw a rectangle over each ROW of parking spaces.</b><br>
    The system reads the parking line markings (white lines painted on the road) inside
    your rectangle to count the number of individual slots automatically.<br><br>
    Total slots = vacant slots + occupied slots (both counted from road lines).
    </div>
    <div class="wbox"><b>⚠️ Do NOT draw one box over the whole parking lot.</b>
    Draw separately over each row. The road lines inside will be detected.
    </div>""",unsafe_allow_html=True)

    canvas_html=f"""<!DOCTYPE html><html><head>
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;font-family:Inter,sans-serif;}}
    body{{background:#f4f6fb;padding:6px;}}
    canvas{{border:2px solid #1565c0;border-radius:12px;cursor:crosshair;
        display:block;max-width:100%;box-shadow:0 3px 14px rgba(21,101,192,.2);}}
    .row{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center;}}
    button{{border:none;border-radius:8px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer;}}
    #bU{{background:#ef5350;color:#fff;}}
    #bC{{background:#78909c;color:#fff;}}
    #bS{{background:#1b5e20;color:#fff;font-size:14px;padding:10px 22px;}}
    #cnt{{background:#e3f2fd;color:#1565c0;border-radius:999px;padding:4px 14px;
        font-size:13px;font-weight:700;}}
    #msg{{margin-top:8px;font-size:13px;font-weight:600;min-height:18px;color:#1565c0;}}
    .tip{{background:#fff8e1;border:1.5px solid #ffa000;border-radius:8px;
        padding:8px 10px;font-size:12px;color:#e65100;margin-top:8px;line-height:1.7;}}
    #slist{{margin-top:8px;font-size:11.5px;color:#555;line-height:2;
        max-height:120px;overflow-y:auto;}}
    #jbox{{display:none;margin-top:10px;background:#e8f5e9;border-radius:10px;
        padding:10px;border:1.5px solid #a5d6a7;}}
    #jbox p{{font-size:12px;color:#1b5e20;font-weight:600;margin-bottom:5px;}}
    textarea{{width:100%;font-size:11px;font-family:monospace;
        border:1px solid #a5d6a7;border-radius:6px;padding:6px;resize:none;background:#fff;}}
    #bcopy{{background:#1565c0;color:#fff;margin-top:6px;width:100%;border-radius:8px;
        padding:7px;font-size:13px;font-weight:600;border:none;cursor:pointer;}}
    </style></head><body>
    <canvas id="C" width="{cw}" height="{ch}"></canvas>
    <div class="row">
      <button id="bU" onclick="undo()">↩ Undo</button>
      <button id="bC" onclick="clr()">🗑 Clear</button>
      <button id="bS" onclick="save()">✅ Save &amp; Continue</button>
      <span id="cnt">0 zones</span>
    </div>
    <div id="msg"></div>
    <div class="tip">
      ✅ Draw over a <b>parking row</b> — system detects slot count from road lines<br>
      ✅ Each white line on the road = one slot boundary
    </div>
    <div id="slist"></div>
    <div id="jbox">
      <p>📋 Redirect blocked — copy JSON &amp; paste in the box below:</p>
      <textarea id="jtxt" rows="3" readonly></textarea>
      <button id="bcopy" onclick="cp()">📋 Copy JSON</button>
    </div>
    <script>
    const C=document.getElementById('C'),ctx=C.getContext('2d');
    const img=new Image(); img.src='data:image/jpeg;base64,{b64}';
    let R=[],drawing=false,sx,sy;
    img.onload=redraw;
    function xy(e){{
      const r=C.getBoundingClientRect();
      return[(e.clientX-r.left)*(C.width/r.width),
             (e.clientY-r.top)*(C.height/r.height)];
    }}
    function redraw(){{
      ctx.clearRect(0,0,C.width,C.height); ctx.drawImage(img,0,0);
      R.forEach((r,i)=>{{
        ctx.fillStyle='rgba(0,230,118,.18)';
        ctx.strokeStyle='#00e676'; ctx.lineWidth=2;
        ctx.fillRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
        ctx.strokeRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
        ctx.fillStyle='rgba(0,0,0,.7)';
        ctx.fillRect(r.x1,r.y1,30,16);
        ctx.fillStyle='#fff'; ctx.font='bold 10px Inter';
        ctx.fillText('Z'+(i+1),r.x1+3,r.y1+12);
      }});
      document.getElementById('cnt').textContent=
        R.length+' zone'+(R.length!=1?'s':'');
      lst();
    }}
    C.addEventListener('mousedown',e=>{{[sx,sy]=xy(e);drawing=true;}});
    C.addEventListener('mousemove',e=>{{
      if(!drawing)return;
      const[cx,cy]=xy(e); redraw();
      ctx.fillStyle='rgba(0,230,118,.18)';
      ctx.strokeStyle='#00e676'; ctx.lineWidth=2;
      ctx.fillRect(sx,sy,cx-sx,cy-sy);
      ctx.strokeRect(sx,sy,cx-sx,cy-sy);
    }});
    C.addEventListener('mouseup',e=>{{
      if(!drawing)return; drawing=false;
      const[cx,cy]=xy(e);
      const x1=Math.round(Math.min(sx,cx)),x2=Math.round(Math.max(sx,cx));
      const y1=Math.round(Math.min(sy,cy)),y2=Math.round(Math.max(sy,cy));
      if(Math.abs(x2-x1)>8&&Math.abs(y2-y1)>8){{
        R.push({{x1,y1,x2,y2}});
        document.getElementById('msg').textContent=
          '✅ Zone '+R.length+' added — slots will be auto-detected from road lines';
      }}
      redraw();
    }});
    function undo(){{if(R.length){{R.pop();redraw();
      document.getElementById('msg').textContent=R.length+' zones';}}}}
    function clr(){{R=[];redraw();document.getElementById('msg').textContent='Cleared';}}
    function lst(){{
      const el=document.getElementById('slist');
      if(!R.length){{el.innerHTML='';return;}}
      el.innerHTML='<b style="font-size:11px;color:#777;">Zones:</b><br>'+
        R.map((r,i)=>`Z${{i+1}}: (${{r.x1}},${{r.y1}})→(${{r.x2}},${{r.y2}}) [${{r.x2-r.x1}}×${{r.y2-r.y1}}px]`).join('<br>');
    }}
    function save(){{
      if(!R.length){{document.getElementById('msg').textContent='⚠️ Draw at least one zone!';return;}}
      const j=JSON.stringify(R);
      document.getElementById('msg').textContent='✅ '+R.length+' zone(s) saved! Redirecting...';
      try{{
        const base=window.parent.location.href.split('?')[0];
        window.parent.location.href=base+'?pvs='+encodeURIComponent(j);
      }}catch(e){{
        document.getElementById('jtxt').value=j;
        document.getElementById('jbox').style.display='block';
        document.getElementById('msg').textContent=
          '✅ '+R.length+' zones! Copy JSON below → paste in Streamlit';
      }}
    }}
    function cp(){{
      const t=document.getElementById('jtxt');
      t.select(); t.setSelectionRange(0,99999);
      navigator.clipboard.writeText(t.value).then(()=>{{
        document.getElementById('bcopy').textContent='✅ Copied!';
        setTimeout(()=>document.getElementById('bcopy').textContent='📋 Copy JSON',2000);
      }}).catch(()=>{{document.execCommand('copy');}});
    }}
    </script></body></html>"""

    c1,c2=st.columns([1.8,1])
    with c1:
        st.components.v1.html(canvas_html,height=ch+280,scrolling=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Drawing Guide</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:10px 12px;
            margin-bottom:10px;font-size:13px;">
        <b style="color:#1b5e20;">✅ Draw over each PARKING ROW</b><br>
        System reads white road lines to count slots:<br>
        <div style="background:#c8e6c9;border:2px solid #43a047;border-radius:6px;
            height:44px;display:flex;align-items:center;justify-content:center;
            gap:2px;margin:6px 0;padding:4px;">
          <div style="border-right:2px solid #1b5e20;flex:1;height:100%;
              display:flex;align-items:center;justify-content:center;
              font-size:10px;font-weight:700;color:#1b5e20;">P1</div>
          <div style="border-right:2px solid #1b5e20;flex:1;height:100%;
              display:flex;align-items:center;justify-content:center;
              font-size:10px;font-weight:700;color:#1b5e20;">P2</div>
          <div style="flex:1;height:100%;display:flex;align-items:center;
              justify-content:center;font-size:10px;font-weight:700;color:#1b5e20;">P3</div>
        </div>
        Each white line found = slot boundary
        </div>
        <div class="ibox" style="font-size:.79rem;">
        <b>If auto-redirect fails:</b><br>
        1. Click ✅ Save &amp; Continue<br>
        2. Copy JSON that appears<br>
        3. Paste below → click Go
        </div>""",unsafe_allow_html=True)
        mj=st.text_area("Paste JSON (if redirect failed):",height=70,
                          placeholder="Paste JSON here...",key="mj")
        if mj and mj.strip():
            try:
                raw=json.loads(mj.strip())
                zones={f"zone_{i}":{"x1":int(float(r["x1"])),"y1":int(float(r["y1"])),
                                     "x2":int(float(r["x2"])),"y2":int(float(r["y2"]))}
                       for i,r in enumerate(raw)}
                st.success(f"✅ {len(zones)} zone(s) loaded!")
                if st.button("➡️  Go to Calibration"):
                    st.session_state.slots=zones
                    st.session_state.stage="calibrate"; st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
        st.markdown('</div>',unsafe_allow_html=True)
        if st.button("← Back"):
            st.session_state.stage="upload"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CALIBRATE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "calibrate":
    topbar()
    stepbar(3)

    zones=st.session_state.slots
    frame=st.session_state.first_frame
    scale=st.session_state.scale

    st.markdown("""
    <div class="ibox"><b>🔧 How calibration works</b><br><br>
    <b>Step 1:</b> Detect parking line markings inside each drawn zone → count individual slots<br>
    <b>Step 2:</b> Read first 60 frames → build median background reference per slot<br>
    <b>Step 3:</b> Any slot that differs from reference = car present = OCCUPIED<br><br>
    Total slots = all slots detected from road lines (occupied + vacant combined)
    </div>""",unsafe_allow_html=True)

    preview_slots=zones_to_slots(zones,frame,scale)
    n=len(preview_slots)

    # Preview
    prev_f=frame.copy()
    fh_p,fw_p=prev_f.shape[:2]
    palette=[(0,200,60),(0,180,220),(220,120,0),(180,0,200),(0,100,200),(200,160,0)]
    for sid,slot in preview_slots.items():
        x1=max(0,slot["x1"]); y1=max(0,slot["y1"])
        x2=min(fw_p,slot["x2"]); y2=min(fh_p,slot["y2"])
        ci=int(sid.split("_")[1])%len(palette)
        cv2.rectangle(prev_f,(x1,y1),(x2,y2),palette[ci],2)
        sn=int(sid.split("_")[1])+1
        (tw,th),_=cv2.getTextSize(f"P{sn}",cv2.FONT_HERSHEY_SIMPLEX,.4,1)
        cv2.rectangle(prev_f,(x1+1,y1+1),(x1+tw+5,y1+th+4),palette[ci],-1)
        cv2.putText(prev_f,f"P{sn}",(x1+3,y1+th+1),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)

    c1,c2=st.columns([1.5,1])
    with c1:
        st.image(cv2.cvtColor(small(prev_f,500),cv2.COLOR_BGR2RGB),
                 caption=f"✅ {n} individual slots detected from {len(zones)} zone(s)",
                 use_column_width=False)
    with c2:
        st.markdown(f"""
        <div class="sbox"><b>✅ {n} parking slots detected</b><br>
        from {len(zones)} zone(s) drawn.<br>
        Total = {n} slots (vacant + occupied combined)
        </div>
        <div class="wbox"><b>Slot count wrong?</b><br>
        Go back and redraw zones more carefully over each parking row.
        Make sure the white road lines are inside your drawn rectangle.
        </div>""",unsafe_allow_html=True)

    if st.button("🚀  Confirm & Run Calibration"):
        prog=st.progress(0,text="Reading frames...")
        cap=cv2.VideoCapture(st.session_state.video_path)
        frames_gray=[]
        for i in range(60):
            ret,f=cap.read()
            if not ret: break
            frames_gray.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32))
            prog.progress((i+1)/60,text=f"Frame {i+1}/60")
        cap.release()
        if frames_gray:
            ref=np.median(frames_gray,axis=0).astype(np.uint8)
            st.session_state.ref_gray=ref
            st.session_state.slots=preview_slots
            st.session_state.events=[]
            st.session_state.occ_history=[]
            st.session_state.dwell_times=[]
            st.session_state.slot_entry_times={}
            st.session_state.max_cars=0
            prog.progress(1.0,text="✅ Done!")
            time.sleep(0.6)
            st.session_state.stage="detect"; st.rerun()
        else:
            st.error("❌ Could not read frames from video.")
    if st.button("← Redraw Zones"):
        st.session_state.stage="draw"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "detect":
    topbar(live=True)
    stepbar(4)

    # ── Summary screen ────────────────────────────────────────────────────────
    if st.session_state.show_summary:
        s=st.session_state.summary
        total=s["total"]; free=s["free"]; occ=s["occ"]
        cars=s["cars"]; peak=s["peak"]
        pct=int(occ/total*100) if total else 0
        dwell=st.session_state.dwell_times
        avg_d=f"{np.mean(dwell):.1f}s" if dwell else "N/A"
        bc="#43a047" if pct<40 else "#ffa000" if pct<75 else "#e53935"
        msg=("🎉 Plenty of parking available!" if pct<40 else
             "⚠️ Parking getting full!" if pct<75 else "🚨 Almost full!")
        mb=("#e8f5e9" if pct<40 else "#fff8e1" if pct<75 else "#ffebee")
        mc=("#1b5e20" if pct<40 else "#e65100" if pct<75 else "#b71c1c")

        st.markdown(f"""
        <div class="sum-wrap">
          <div style="text-align:center;margin-bottom:1.4rem;">
            <div style="font-size:3.5rem;">🅿️</div>
            <div style="font-size:1.55rem;font-weight:800;color:#0a2463;">
              Parking Session Summary</div>
            <div style="font-size:.85rem;color:#78909c;">
              Session: {st.session_state.session_start or 'N/A'}</div>
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
          <div class="pw">
            <div class="pf" style="width:{pct}%;background:{bc};"></div>
          </div>
          <div class="pl" style="margin-bottom:.8rem;">
            <span>🟢 {free} free ({100-pct}%)</span>
            <span>🔴 {occ} occupied ({pct}%)</span>
          </div>
          <div style="background:{mb};color:{mc};border-radius:10px;
              padding:.7rem;text-align:center;font-weight:700;font-size:.95rem;">
            {msg}
          </div>
        </div>""",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        pdf_data,mime,fname=make_report(s,st.session_state.events,
                                         st.session_state.dwell_times,
                                         st.session_state.session_start)
        ext=fname.split('.')[-1].upper()
        st.download_button(f"📥 Download Report ({ext})",
                           data=pdf_data,file_name=fname,mime=mime)
        c1,c2=st.columns(2)
        with c1:
            if st.button("🔄 Detect Again"):
                st.session_state.show_summary=False
                st.session_state.events=[]; st.session_state.occ_history=[]
                st.session_state.dwell_times=[]; st.rerun()
        with c2:
            if st.button("🖊 Redraw Zones"):
                st.session_state.show_summary=False
                st.session_state.stage="draw"; st.rerun()
        st.stop()

    # ── Live detection ────────────────────────────────────────────────────────
    model=load_yolo()
    slots=st.session_state.slots
    ref=st.session_state.ref_gray
    total=len(slots)

    thresh=st.sidebar.slider("🎚 Detection Sensitivity",10,80,35,
        help="Increase if empty slots show FULL. Decrease if occupied shows FREE.")
    st.sidebar.markdown("**Tips:**\n- Start at 35\n- Adjust by 5 at a time")

    # ── LAYOUT: VIDEO LEFT | ALL STATS RIGHT ─────────────────────────────────
    c_vid,c_stat=st.columns([1.15,1])

    with c_vid:
        st.markdown('<div class="card"><div class="card-title">🚗 Live Detection Feed</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;gap:1rem;align-items:center;font-size:.77rem;
            color:#424242;background:#f5f7ff;border-radius:8px;
            padding:.45rem .9rem;margin-bottom:.7rem;border:1px solid #e3eaf7;flex-wrap:wrap;">
          <span>🟢 FREE = vacant</span>
          <span>🔴 FULL = occupied</span>
          <span style="color:#888;">🔵 = YOLO car detection</span>
        </div>""",unsafe_allow_html=True)
        vid_ph=st.empty()
        alert_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

    with c_stat:
        # Stats card
        st.markdown('<div class="card"><div class="card-title">📊 Live Statistics</div>',
                    unsafe_allow_html=True)
        stat_ph=st.empty()
        prog_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

        # Analytics card
        st.markdown('<div class="card"><div class="card-title">📈 Analytics</div>',
                    unsafe_allow_html=True)
        an_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

        # Events card
        st.markdown('<div class="card"><div class="card-title">📋 Live Events</div>',
                    unsafe_allow_html=True)
        ev_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

        # Controls card
        st.markdown('<div class="card"><div class="card-title">⚙️ Controls</div>',
                    unsafe_allow_html=True)
        stop_btn=st.button("⏹  Stop & Show Summary")
        st.markdown("<div style='height:.3rem'></div>",unsafe_allow_html=True)
        if st.button("← Redraw Zones"):
            st.session_state.stage="draw"; st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

    # ── Detection loop ────────────────────────────────────────────────────────
    cap=cv2.VideoCapture(st.session_state.video_path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    fc=0; boxes=[]; prev={sid:False for sid in slots}
    alert_state=None; max_cars=0

    while cap.isOpened() and not stop_btn:
        ret,frame=cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        fc+=1; fh,fw=frame.shape[:2]

        # YOLO every 4 frames
        if fc%4==0:
            try:
                res=model.predict(frame,classes=[2,3,5,7],
                                  conf=0.25,iou=0.45,verbose=False)
                boxes=res[0].boxes.xyxy.cpu().numpy().tolist() if res[0].boxes else []
            except: boxes=[]

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Update each slot
        for sid,slot in slots.items():
            now=update_slot_state(slot,gray,ref,boxes,fw,fh,thresh)
            if now!=prev[sid]:
                n=int(sid.split("_")[1])+1
                ts=datetime.now().strftime("%H:%M:%S")
                if now:
                    st.session_state.events.append(
                        f"🔴 {ts} — P{n} OCCUPIED (car entered)")
                    st.session_state.slot_entry_times[sid]=datetime.now()
                else:
                    st.session_state.events.append(
                        f"🟢 {ts} — P{n} VACANT (car left)")
                    if sid in st.session_state.slot_entry_times:
                        dwell=(datetime.now()-st.session_state.slot_entry_times[sid]).total_seconds()
                        st.session_state.dwell_times.append(dwell)
                        del st.session_state.slot_entry_times[sid]
            prev[sid]=now

        # Draw
        ann=draw_detection(frame.copy(),slots,boxes)
        vid_ph.image(cv2.cvtColor(small(ann,460),cv2.COLOR_BGR2RGB),use_column_width=False)

        # Counts
        occ=sum(1 for s in slots.values() if s["occupied"])
        free=total-occ; pct=int(occ/total*100) if total else 0
        cars=len(boxes); max_cars=max(max_cars,cars)
        st.session_state.occ_history.append(occ)
        bc="#e53935" if pct>70 else "#ffa000" if pct>40 else "#43a047"

        stat_ph.markdown(f"""
        <div class="srow">
          <div class="sc sc-t"><div class="sc-n">{total}</div><div class="sc-l">Total</div></div>
          <div class="sc sc-v"><div class="sc-n">{free}</div><div class="sc-l">🟢 Vacant</div></div>
          <div class="sc sc-o"><div class="sc-n">{occ}</div><div class="sc-l">🔴 Occupied</div></div>
          <div class="sc sc-c"><div class="sc-n">{cars}</div><div class="sc-l">🚗 Cars</div></div>
        </div>""",unsafe_allow_html=True)

        prog_ph.markdown(f"""
        <div class="pw">
          <div class="pf" style="width:{pct}%;background:{bc};"></div>
        </div>
        <div class="pl">
          <span>🟢 {free} free ({100-pct}%)</span>
          <span>🔴 {occ} taken ({pct}%)</span>
        </div>""",unsafe_allow_html=True)

        dwell_list=st.session_state.dwell_times
        avg_d=f"{np.mean(dwell_list):.0f}s" if dwell_list else "—"
        peak=max(st.session_state.occ_history) if st.session_state.occ_history else 0
        an_ph.markdown(f"""
        <div class="an-row">
          <div class="an-chip"><div class="n">{peak}</div><div class="l">Peak Occ.</div></div>
          <div class="an-chip"><div class="n">{avg_d}</div><div class="l">Avg Dwell</div></div>
          <div class="an-chip"><div class="n">{len(st.session_state.events)}</div>
              <div class="l">Events</div></div>
        </div>""",unsafe_allow_html=True)

        if pct>=95:
            alert_ph.markdown(
                '<div class="alert-full">🚨 LOT FULL — Redirect incoming vehicles!</div>',
                unsafe_allow_html=True)
            alert_state="full"
        elif pct<70 and alert_state=="full":
            alert_ph.markdown(
                '<div class="alert-ok">✅ Spaces available now!</div>',
                unsafe_allow_html=True)
            alert_state=None

        if st.session_state.events:
            evs="".join(f"<div>{e}</div>" for e in st.session_state.events[-6:])
            ev_ph.markdown(f'<div class="ev-box">{evs}</div>',unsafe_allow_html=True)

        time.sleep(1/fps)

    cap.release()
    occ_f=sum(1 for s in slots.values() if s["occupied"])
    peak_f=max(st.session_state.occ_history) if st.session_state.occ_history else 0
    st.session_state.summary={
        "total":total,"free":total-occ_f,"occ":occ_f,
        "cars":max_cars,"peak":peak_f
    }
    st.session_state.show_summary=True
    st.rerun()