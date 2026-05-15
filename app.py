def draw_result(frame, slots, boxes, blink_phase=0):
    for box in boxes:
        bx1, by1, bx2, by2 = [int(v) for v in box[:4]]

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 255), 2)

    for sid, slot in slots.items():
        x1 = slot["x1"]
        y1 = slot["y1"]
        x2 = slot["x2"]
        y2 = slot["y2"]

        occ = slot.get("occupied", False)

        if occ:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 70), 3)

    return frame
