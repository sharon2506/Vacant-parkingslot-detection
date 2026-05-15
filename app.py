def check_slot_bg(frame_gray, ref_max, ref_min, slot):
    x1 = max(0, slot["x1"])
    y1 = max(0, slot["y1"])
    x2 = min(frame_gray.shape[1], slot["x2"])
    y2 = min(frame_gray.shape[0], slot["y2"])

    cur = cv2.GaussianBlur(frame_gray[y1:y2, x1:x2], (5,5), 0)

    score = np.mean(cur)

    return float(score)


def check_slot_yolo(slot, boxes, fw, fh):
    x1 = slot["x1"]
    y1 = slot["y1"]
    x2 = slot["x2"]
    y2 = slot["y2"]

    slot_area = max(1, (x2 - x1) * (y2 - y1))

    for box in boxes:
        bx1, by1, bx2, by2 = [float(v) for v in box[:4]]

        ix1 = max(bx1, x1)
        iy1 = max(by1, y1)
        ix2 = min(bx2, x2)
        iy2 = min(by2, y2)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        frac_of_slot = inter / slot_area

        if frac_of_slot >= 0.08:
            return True, frac_of_slot

    return False, 0.0
