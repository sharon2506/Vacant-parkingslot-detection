def stepbar(cur):
    labels = ["Upload", "Draw Slots", "Calibrate", "Detect"]
    h = '<div class="stepbar">'
    for i, l in enumerate(labels, 1):
        cls  = "done" if i < cur else ("active" if i == cur else "stp")
        icon = "✓" if i < cur else str(i)
        h += f'<div class="stp {cls}"><div class="stp-n">{icon}</div>{l}</div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)


def show_topbar(live=False):
    pill = '<div class="live-pill">● LIVE</div>' if live else ''
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-icon">🚗</div>
      <div>
        <p class="topbar-title">ParkVision Pro</p>
      </div>
      {pill}
    </div>""", unsafe_allow_html=True)
