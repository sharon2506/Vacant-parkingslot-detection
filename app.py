import streamlit as st
import cv2
import json
import numpy as np
import tempfile
import os
import time
import base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ParkVision", layout="wide",
                   initial_sidebar_state="collapsed")

SLOTS_FILE = "drawn_slots.json"   # canvas saves here; detection reads from here

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main,.main .block-container {
    background:#f8faff !important; font-family:'Inter',sans-serif !important; }
#MainMenu,footer,header,
[data-testid="stToolbar"],[data-testid="stDecoration"]{ visibility:hidden; }
.block-container{ padding:1.8rem 2.5rem !important; max-width:1400px; }

.topbar{ background:linear-gradient(135deg,#1565c0,#0d47a1); border-radius:18px;
    padding:1.5rem 2rem; display:flex; align-items:center; gap:1rem;
    margin-bottom:1.8rem; box-shadow:0 4px 20px rgba(21,101,192,.3); }
.topbar-icon{ font-size:2.4rem; }
.topbar-title{ font-size:1.75rem; font-weight:700; color:#fff; margin:0; }
.topbar-sub{ font-size:.82rem; color:#90caf9; margin:.2rem 0 0; }

.stepbar{ display:flex; align-items:center; gap:.4rem; margin-bottom:1.6rem; }
.stp{ display:flex; align-items:center; gap:.4rem; font-size:.8rem; font-weight:600; color:#9e9e9e; }
.stp.active{ color:#1565c0; } .stp.done{ color:#2e7d32; }
.stp-n{ width:26px; height:26px; border-radius:50%; font-size:.72rem; font-weight:700;
    display:flex; align-items:center; justify-content:center;
    background:#e0e0e0; color:#757575; }
.stp.active .stp-n{ background:#1565c0; color:#fff; }
.stp.done   .stp-n{ background:#2e7d32; color:#fff; }
.stp-line{ flex:1; height:2px; background:#e0e0e0; border-radius:2px; }

.card{ background:#fff; border:1.5px solid #e3eaf7; border-radius:14px;
    padding:1.3rem 1.4rem; margin-bottom:1rem;
    box-shadow:0 2px 10px rgba(0,0,0,.04); }
.card-title{ font-size:.72rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#1565c0; margin-bottom:.9rem; }

.srow{ display:flex; gap:.9rem; margin-bottom:1.2rem; }
.sc{ flex:1; border-radius:14px; padding:1.2rem .8rem; text-align:center; }
.sc-t{ background:#e8eaf6; border:1.5px solid #c5cae9; }
.sc-v{ background:#e8f5e9; border:1.5px solid #a5d6a7; }
.sc-o{ background:#ffebee; border:1.5px solid #ef9a9a; }
.sc-n{ font-size:2.8rem; font-weight:700; line-height:1; margin-bottom:.2rem; }
.sc-t .sc-n{ color:#283593; } .sc-v .sc-n{ color:#1b5e20; } .sc-o .sc-n{ color:#b71c1c; }
.sc-l{ font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.sc-t .sc-l{ color:#7986cb; } .sc-v .sc-l{ color:#66bb6a; } .sc-o .sc-l{ color:#ef5350; }

.pw{ background:#f0f0f0; border-radius:999px; height:10px; overflow:hidden;
    margin:.7rem 0 .3rem; border:1px solid #e0e0e0; }
.pf{ height:100%; border-radius:999px; background:linear-gradient(90deg,#ef5350,#b71c1c); }
.pl{ display:flex; justify-content:space-between; font-size:.73rem; color:#9e9e9e; font-weight:500; }

.ibox{ background:#e3f2fd; border-left:4px solid #1565c0; border-radius:0 10px 10px 0;
    padding:.85rem 1rem; font-size:.84rem; color:#0d47a1; margin-bottom:.9rem; line-height:1.8; }
.wbox{ background:#fff8e1; border-left:4px solid #f9a825; border-radius:0 10px 10px 0;
    padding:.85rem 1rem; font-size:.84rem; color:#e65100; margin-bottom:.9rem; line-height:1.8; }
.sbox{ background:#e8f5e9; border-left:4px solid #43a047; border-radius:0 10px 10px 0;
    padding:.85rem 1rem; font-size:.84rem; color:#1b5e20; margin-bottom:.9rem; line-height:1.8; }

.stButton>button{ background:linear-gradient(135deg,#1565c0,#0d47a1) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:.87rem !important; width:100% !important; padding:.62rem 1.4rem !important; }
.stButton>button:hover{ opacity:.88 !important; }

.leg{ display:flex; gap:1.4rem; align-items:center; font-size:.8rem; color:#424242;
    font-weight:500; background:#f5f5f5; border-radius:8px;
    padding:.55rem .9rem; margin-bottom:.8rem; border:1px solid #e0e0e0; }
.dot{ width:13px; height:13px; border-radius:3px; display:inline-block;
    margin-right:4px; vertical-align:middle; }
.dg{ background:#43a047; } .dr{ background:#e53935; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="topbar-icon">🅿️</div>
  <div>
    <p class="topbar-title">ParkVision</p>
    <p class="topbar-sub">AI-powered vacant parking slot detector · YOLOv8 Segmentation</p>
  </div>
</div>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in dict(slots={}, video_path=None, first_frame=None,
                stage="upload", canvas_scale=1.0).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Step bar ──────────────────────────────────────────────────────────────────
def stepbar(cur):
    labels = ["Upload Video","Draw Slots","Live Detection"]
    h = '<div class="stepbar">'
    for i,l in enumerate(labels,1):
        cls  = "done" if i<cur else ("active" if i==cur else "stp")
        icon = "✓"    if i<cur else str(i)
        h += f'<div class="stp {cls}"><div class="stp-n">{icon}</div>{l}</div>'
        if i<len(labels): h += '<div class="stp-line"></div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)

# ── YOLO segmentation ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n-seg.pt")

# ── Detection helpers ─────────────────────────────────────────────────────────
def get_detections(frame, model):
    results = model(frame, classes=[2,3,5,7], conf=0.25, verbose=False)
    r = results[0]
    boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
    dets  = []
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        for i,box in enumerate(boxes):
            dets.append((box, masks[i] if i<len(masks) else None))
    else:
        for box in boxes:
            dets.append((box, None))
    return dets

def mask_overlap(mask, x1,y1,x2,y2, fw,fh):
    if mask is None: return 0.0
    m = cv2.resize(mask.astype(np.uint8),(fw,fh),interpolation=cv2.INTER_NEAREST)
    region = m[y1:y2, x1:x2]
    total  = max(1,(y2-y1)*(x2-x1))
    return region.sum() / total

def box_overlap(box, x1,y1,x2,y2):
    bx1,by1,bx2,by2 = box
    ix1,iy1 = max(bx1,x1),max(by1,y1)
    ix2,iy2 = min(bx2,x2),min(by2,y2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    area  = max(1,(x2-x1)*(y2-y1))
    return inter/area

def center_inside(box, x1,y1,x2,y2):
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
    return x1<cx<x2 and y1<cy<y2

def update_occupancy(dets, slots, sx, sy, fw, fh):
    for s in slots.values():
        s["occupied"] = False
    for (box, mask) in dets:
        for slot in slots.values():
            # Convert canvas coords → frame coords
            fx1=int(slot["x1"]*sx); fy1=int(slot["y1"]*sy)
            fx2=int(slot["x2"]*sx); fy2=int(slot["y2"]*sy)
            fx1=max(0,fx1); fy1=max(0,fy1)
            fx2=min(fw,fx2); fy2=min(fh,fy2)

            occ = False
            if mask is not None:
                if mask_overlap(mask,fx1,fy1,fx2,fy2,fw,fh) > 0.10:
                    occ = True
            if not occ and center_inside(box,fx1,fy1,fx2,fy2):
                occ = True
            if not occ and box_overlap(box,fx1,fy1,fx2,fy2) > 0.15:
                occ = True
            if occ:
                slot["occupied"] = True

def draw_on_frame(frame, slots, dets, sx, sy):
    fh,fw = frame.shape[:2]
    # Draw vehicle masks
    for (box,mask) in dets:
        if mask is not None:
            m = cv2.resize(mask.astype(np.uint8),(fw,fh),interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(frame)
            overlay[m==1] = (0,200,255)
            cv2.addWeighted(overlay,0.35,frame,1.0,0,frame)
    # Draw slots
    for sid,slot in slots.items():
        x1=int(slot["x1"]*sx); y1=int(slot["y1"]*sy)
        x2=int(slot["x2"]*sx); y2=int(slot["y2"]*sy)
        occ = slot.get("occupied",False)
        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(30,30,220) if occ else (30,200,70),-1)
        cv2.addWeighted(ov,.28,frame,.72,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(10,10,180) if occ else (10,160,40),2)
        cv2.putText(frame,"FULL" if occ else "FREE",(x1+4,y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    return frame

def frame_to_b64(frame, max_w=680):
    h,w = frame.shape[:2]
    if w>max_w:
        s=max_w/w; frame=cv2.resize(frame,(int(w*s),int(h*s)))
    _,buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,90])
    return base64.b64encode(buf).decode(), frame.shape[1], frame.shape[0]

def resize_frame(frame, max_w=460):
    h,w = frame.shape[:2]
    if w>max_w:
        s=max_w/w; frame=cv2.resize(frame,(int(w*s),int(h*s)))
    return frame

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    stepbar(1)
    c1,c2 = st.columns([1.5,1])

    with c1:
        st.markdown('<div class="card"><div class="card-title">Upload Parking Lot Video</div>', unsafe_allow_html=True)
        up = st.file_uploader("",type=["mp4","avi","mov","mkv"],label_visibility="collapsed")
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
            tmp.write(up.read()); tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            ret,frame = cap.read(); cap.release()
            if ret and frame is not None:
                st.session_state.video_path  = tmp.name
                st.session_state.first_frame = frame.copy()
                st.image(cv2.cvtColor(resize_frame(frame,480),cv2.COLOR_BGR2RGB),
                         caption="✅ Video loaded",use_column_width=False)
                st.markdown("<br>",unsafe_allow_html=True)
                if st.button("➡️  Next: Draw Parking Slots"):
                    st.session_state.stage="draw"; st.rerun()
            else:
                st.error("❌ Cannot read video. Run `python3 convert_video.py` first.")
        st.markdown('</div>',unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ibox"><strong>📋 How it works</strong><br><br>
        1️⃣ Upload parking lot video<br>
        2️⃣ Draw a small rectangle on <b>each individual</b> parking slot<br>
        3️⃣ YOLOv8 Segmentation detects vehicles with pixel masks<br>
        4️⃣ See 🟢 vacant · 🔴 occupied live
        </div>
        <div class="wbox"><strong>⚠️ Video not loading?</strong><br><br>
        <code>python3 convert_video.py</code><br>Then upload <b>output.mp4</b>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DRAW SLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "draw":
    stepbar(2)

    frame = st.session_state.first_frame
    CANVAS_W = 680
    h_orig,w_orig = frame.shape[:2]
    scale    = CANVAS_W / w_orig
    canvas_h = int(h_orig * scale)
    st.session_state.canvas_scale = scale

    b64,cw,ch = frame_to_b64(frame, max_w=CANVAS_W)

    st.markdown("""
    <div class="wbox">
    <strong>⚠️ Important — Draw SMALL rectangles on EACH parking slot individually!</strong><br>
    Do NOT draw one big rectangle over the whole parking lot.<br>
    Each rectangle = exactly one parking space. See the example below.
    </div>
    <div class="ibox">
    <strong>✏️ How to draw:</strong> Click and drag a small rectangle over one parking space at a time.
    Use <b>Undo Last</b> to remove mistakes. When all slots are drawn click <b>✅ Save & Detect</b>.
    The app will <b>automatically</b> move to detection — no copy-paste needed!
    </div>
    """, unsafe_allow_html=True)

    # ── HTML5 Canvas — saves to file, Streamlit polls for it ────────────────
    canvas_html = f"""<!DOCTYPE html><html><head>
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;font-family:Inter,sans-serif;}}
    body{{background:#f8faff;padding:8px;}}
    h4{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
        color:#1565c0;margin-bottom:6px;}}
    canvas{{border:2px solid #1565c0;border-radius:10px;cursor:crosshair;
        display:block;max-width:100%;}}
    .row{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center;}}
    button{{border:none;border-radius:8px;padding:8px 14px;
        font-size:13px;font-weight:600;cursor:pointer;}}
    #bUndo{{background:#ef5350;color:#fff;}}
    #bClear{{background:#757575;color:#fff;}}
    #bSave{{background:#2e7d32;color:#fff;font-size:14px;padding:10px 20px;}}
    #count{{font-size:13px;font-weight:600;color:#1565c0;margin-left:4px;}}
    #msg{{margin-top:8px;font-size:13px;font-weight:600;min-height:20px;}}
    #slotList{{margin-top:10px;font-size:12px;color:#333;line-height:2;
        max-height:200px;overflow-y:auto;}}
    .tip{{background:#fff8e1;border:1px solid #f9a825;border-radius:8px;
        padding:8px 10px;font-size:12px;color:#e65100;margin-top:8px;line-height:1.6;}}
    </style></head><body>
    <h4>🖱 Click &amp; drag a small rectangle over each parking space</h4>
    <canvas id="C" width="{cw}" height="{ch}"></canvas>
    <div class="row">
      <button id="bUndo" onclick="undoLast()">↩ Undo Last</button>
      <button id="bClear" onclick="clearAll()">🗑 Clear All</button>
      <button id="bSave" onclick="saveSlots()">✅ Save &amp; Detect</button>
      <span id="count">0 slots</span>
    </div>
    <div id="msg"></div>
    <div class="tip">
      ⚠️ Draw <b>one small rectangle per parking space</b>.<br>
      Don't draw over the whole lot — draw each space individually!
    </div>
    <div id="slotList"></div>

    <script>
    const C   = document.getElementById('C');
    const ctx = C.getContext('2d');
    const img = new Image();
    img.src   = 'data:image/jpeg;base64,{b64}';
    let rects=[],drawing=false,sx,sy;

    img.onload = redraw;

    function getXY(e){{
      const r=C.getBoundingClientRect();
      return [(e.clientX-r.left)*(C.width/r.width),
              (e.clientY-r.top)*(C.height/r.height)];
    }}

    function redraw(){{
      ctx.clearRect(0,0,C.width,C.height);
      ctx.drawImage(img,0,0);
      rects.forEach((r,i)=>{{
        ctx.fillStyle='rgba(50,220,80,.22)';
        ctx.strokeStyle='#00e676';
        ctx.lineWidth=2;
        ctx.fillRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
        ctx.strokeRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);
        ctx.fillStyle='#fff';
        ctx.font='bold 11px Inter,sans-serif';
        ctx.fillText('P'+(i+1),r.x1+3,r.y1+13);
      }});
      document.getElementById('count').textContent=rects.length+' slot'+(rects.length!=1?'s':'');
      updateList();
    }}

    C.addEventListener('mousedown',e=>{{[sx,sy]=getXY(e);drawing=true;}});

    C.addEventListener('mousemove',e=>{{
      if(!drawing)return;
      const[cx,cy]=getXY(e);
      redraw();
      ctx.fillStyle='rgba(50,220,80,.22)';
      ctx.strokeStyle='#00e676';
      ctx.lineWidth=2;
      ctx.fillRect(sx,sy,cx-sx,cy-sy);
      ctx.strokeRect(sx,sy,cx-sx,cy-sy);
    }});

    C.addEventListener('mouseup',e=>{{
      if(!drawing)return; drawing=false;
      const[cx,cy]=getXY(e);
      const x1=Math.round(Math.min(sx,cx)),x2=Math.round(Math.max(sx,cx));
      const y1=Math.round(Math.min(sy,cy)),y2=Math.round(Math.max(sy,cy));
      if(Math.abs(x2-x1)>8&&Math.abs(y2-y1)>8){{
        rects.push({{x1,y1,x2,y2}});
        document.getElementById('msg').style.color='#2e7d32';
        document.getElementById('msg').textContent='✅ Slot P'+rects.length+' added!';
      }}
      redraw();
    }});

    function undoLast(){{
      if(rects.length){{rects.pop();redraw();
        document.getElementById('msg').textContent=rects.length+' slots remaining';}}
    }}
    function clearAll(){{
      rects=[];redraw();
      document.getElementById('msg').textContent='Cleared!';
    }}
    function updateList(){{
      const el=document.getElementById('slotList');
      if(!rects.length){{el.innerHTML='';return;}}
      el.innerHTML='<b style="font-size:11px;color:#555;">Drawn slots:</b><br>'+
        rects.map((r,i)=>`P${{i+1}}: (${{r.x1}},${{r.y1}})→(${{r.x2}},${{r.y2}})`).join('<br>');
    }}

    function saveSlots(){{
      if(!rects.length){{
        document.getElementById('msg').style.color='#c62828';
        document.getElementById('msg').textContent='⚠️ Draw at least one slot first!';
        return;
      }}
      // Post JSON to Streamlit via the hidden form trick
      const json=JSON.stringify(rects);
      // Write to localStorage so Streamlit can pick it up via st.components
      try{{localStorage.setItem('parkvision_slots',json);}}catch(e){{}}
      // Also send via URL param
      try{{
        const base=window.parent.location.href.split('?')[0];
        window.parent.location.href=base+'?pv_slots='+encodeURIComponent(json);
      }}catch(e){{
        document.getElementById('msg').style.color='#1565c0';
        document.getElementById('msg').textContent=
          '✅ '+rects.length+' slots ready! Scroll down and click Confirm Slots button.';
        // Fallback: show the JSON for manual paste
        const out=document.createElement('textarea');
        out.value=json; out.style.cssText='width:100%;margin-top:8px;font-size:11px;height:60px;';
        out.id='fallbackJSON';
        const existing=document.getElementById('fallbackJSON');
        if(existing)existing.replaceWith(out); else document.body.appendChild(out);
        out.select();
      }}
    }}
    </script></body></html>"""

    # Check if slots arrived via URL param
    qp = st.query_params
    auto_slots = None
    if "pv_slots" in qp:
        try:
            raw = json.loads(qp["pv_slots"])
            auto_slots = raw
        except:
            pass

    if auto_slots:
        # Auto-received slots — save and go to detect
        parsed = {}
        for i,r in enumerate(auto_slots):
            parsed[f"slot_{i}"] = {
                "x1":int(float(r["x1"])),"y1":int(float(r["y1"])),
                "x2":int(float(r["x2"])),"y2":int(float(r["y2"])),
                "occupied":False
            }
        st.session_state.slots = parsed
        st.session_state.stage = "detect"
        st.query_params.clear()
        st.rerun()

    c1,c2 = st.columns([1.8,1])

    with c1:
        st.components.v1.html(canvas_html, height=ch+260, scrolling=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">How to Draw Slots Correctly</div>', unsafe_allow_html=True)

        # Visual guide using ASCII / colored boxes
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:12px;margin-bottom:12px;font-size:13px;color:#e65100;">
        <b>❌ WRONG — One big rectangle:</b><br>
        <div style="background:#ffcdd2;border:2px solid red;border-radius:4px;
            padding:20px;margin:6px 0;text-align:center;font-size:11px;color:#b71c1c;">
          One giant slot over everything
        </div>
        </div>
        <div style="background:#e8f5e9;border-radius:10px;padding:12px;margin-bottom:12px;font-size:13px;color:#2e7d32;">
        <b>✅ CORRECT — One small rectangle per space:</b><br>
        <div style="display:flex;gap:4px;margin:6px 0;flex-wrap:wrap;">
          <div style="background:#c8e6c9;border:2px solid #43a047;border-radius:3px;
              padding:6px 8px;font-size:10px;color:#1b5e20;font-weight:bold;">P1</div>
          <div style="background:#c8e6c9;border:2px solid #43a047;border-radius:3px;
              padding:6px 8px;font-size:10px;color:#1b5e20;font-weight:bold;">P2</div>
          <div style="background:#c8e6c9;border:2px solid #43a047;border-radius:3px;
              padding:6px 8px;font-size:10px;color:#1b5e20;font-weight:bold;">P3</div>
          <div style="background:#c8e6c9;border:2px solid #43a047;border-radius:3px;
              padding:6px 8px;font-size:10px;color:#1b5e20;font-weight:bold;">P4</div>
        </div>
        Each box = one parking space
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ibox">
        <b>Steps:</b><br>
        1. Zoom in to see each parking space clearly<br>
        2. Draw a small rectangle over <b>one</b> space<br>
        3. Repeat for every space<br>
        4. Click <b>✅ Save &amp; Detect</b>
        </div>
        """, unsafe_allow_html=True)

        # Manual confirm button (fallback if URL redirect blocked)
        st.markdown("---")
        st.markdown("**If the page didn't move automatically, paste the JSON here:**")
        manual_json = st.text_area("JSON:", height=80, key="manual_json",
                                   placeholder='Paste JSON here as fallback...')

        if manual_json and manual_json.strip():
            try:
                raw    = json.loads(manual_json.strip())
                parsed = {}
                for i,r in enumerate(raw):
                    parsed[f"slot_{i}"] = {
                        "x1":int(float(r["x1"])),"y1":int(float(r["y1"])),
                        "x2":int(float(r["x2"])),"y2":int(float(r["y2"])),
                        "occupied":False
                    }
                st.success(f"✅ {len(parsed)} slots loaded!")
                if st.button("🚀  Start Detection"):
                    st.session_state.slots = parsed
                    st.session_state.stage = "detect"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("← Back to Upload"):
            st.session_state.stage="upload"
            st.session_state.slots={}
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "detect":
    stepbar(3)

    model = load_model()
    slots = st.session_state.slots
    total = len(slots)
    scale = st.session_state.canvas_scale
    sx    = 1.0 / scale
    sy    = 1.0 / scale

    c_vid,c_stat = st.columns([1.1,1])

    with c_vid:
        st.markdown('<div class="card"><div class="card-title">📹 Live Video Feed</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="leg">
          <span><span class="dot dg"></span>Vacant</span>
          <span><span class="dot dr"></span>Occupied</span>
          <span style="font-size:.75rem;color:#888;">Yellow = detected vehicle</span>
        </div>""", unsafe_allow_html=True)
        vid_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with c_stat:
        st.markdown('<div class="card"><div class="card-title">📊 Live Statistics</div>', unsafe_allow_html=True)
        stat_ph = st.empty()
        prog_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sbox" style="font-size:.8rem;">
        🔬 <b>YOLOv8 Segmentation</b> active<br>
        Pixel-level vehicle masks for precise detection
        </div>""", unsafe_allow_html=True)
        stop_btn = st.button("⏹  Stop Detection")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Redraw Slots"):
            st.session_state.stage="draw"; st.rerun()

    cap  = cv2.VideoCapture(st.session_state.video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    fc   = 0
    last_dets = []

    while cap.isOpened() and not stop_btn:
        ret,frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue

        fc += 1
        fh,fw = frame.shape[:2]

        if fc % 3 == 0:
            last_dets = get_detections(frame, model)

        update_occupancy(last_dets, slots, sx, sy, fw, fh)
        ann  = draw_on_frame(frame.copy(), slots, last_dets, sx, sy)
        disp = resize_frame(ann, 460)
        vid_ph.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB), use_column_width=False)

        occ  = sum(1 for s in slots.values() if s.get("occupied",False))
        free = total - occ
        pct  = int(occ/total*100) if total else 0

        stat_ph.markdown(f"""
        <div class="srow">
          <div class="sc sc-t"><div class="sc-n">{total}</div><div class="sc-l">Total</div></div>
          <div class="sc sc-v"><div class="sc-n">{free}</div><div class="sc-l">Vacant</div></div>
          <div class="sc sc-o"><div class="sc-n">{occ}</div><div class="sc-l">Occupied</div></div>
        </div>""", unsafe_allow_html=True)

        prog_ph.markdown(f"""
        <div class="pw"><div class="pf" style="width:{pct}%"></div></div>
        <div class="pl">
          <span>🟢 {free} free</span><span>{pct}% occupied</span><span>🔴 {occ} taken</span>
        </div>""", unsafe_allow_html=True)

        time.sleep(1/fps)

    cap.release()
    if stop_btn:
        st.success("✅ Detection stopped.")
        