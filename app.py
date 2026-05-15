elif st.session_state.stage == "upload":

    up = st.file_uploader(
        "",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        tmp.write(up.read())

        cap = cv2.VideoCapture(tmp.name)

        ret, frame = cap.read()

        if ret:
            st.session_state.video_path = tmp.name
            st.session_state.first_frame = frame.copy()
