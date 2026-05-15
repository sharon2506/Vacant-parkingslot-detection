for k, v in dict(
    slots={}, video_path=None, first_frame=None,
    ref_frame=None, stage="splash", scale=1.0,
    show_summary=False, summary={}, calib_done=False,
    events=[], slot_entry_times={}, dwell_times=[],
    occ_history=[], max_cars_seen=0, session_start=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v
