elif st.session_state.stage == "detect":

    model = load_model()

    slots = st.session_state.slots

    cap = cv2.VideoCapture(st.session_state.video_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        boxes = detect_vehicles_yolo(frame, model)

        for sid, slot in slots.items():

            yolo_occ, overlap = check_slot_yolo(
                slot,
                boxes,
                frame.shape[1],
                frame.shape[0]
            )

            slot["occupied"] = yolo_occ

        
