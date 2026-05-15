elif st.session_state.stage == "calibrate":

    zones = st.session_state.slots
    frame = st.session_state.first_frame

    preview_slots = zones_to_slots(zones, frame, scale)

    if st.button("🔧 Run Calibration & Start Detection"):

        cap = cv2.VideoCapture(st.session_state.video_path)

        frames_gray = []

        while len(frames_gray) < 120:
            ret, f = cap.read()

            if not ret:
                break

            frames_gray.append(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            )
