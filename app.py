def find_peaks(signal, dim, min_dist_frac=0.06):
    if len(signal) == 0:
        return []

    mean = signal.mean()
    std  = signal.std()
    thr  = mean + 0.7 * std

    peaks = []
    for i in range(len(signal)):
        if signal[i] > thr:
            peaks.append(i)

    return peaks


def make_slot(x1, y1, x2, y2):
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "occupied": False,
        "score": 0.0,
        "ema": 0.0
    }


def detect_slots_in_zone(frame, zone, scale):
    sx = 1.0 / scale

    zx1 = max(0, int(zone["x1"] * sx))
    zy1 = max(0, int(zone["y1"] * sx))
    zx2 = min(frame.shape[1], int(zone["x2"] * sx))
    zy2 = min(frame.shape[0], int(zone["y2"] * sx))

    roi  = frame[zy1:zy2, zx1:zx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    sobelx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_proj = np.abs(sobelx).mean(axis=0)

    slots = []

    aspect = (zx2 - zx1) / max((zy2 - zy1), 1)

    if aspect >= 1.6:
        n = max(1, round(aspect / 1.5))
        sw = (zx2 - zx1) // n

        for i in range(n):
            slots.append(make_slot(
                zx1 + i * sw,
                zy1,
                zx1 + (i + 1) * sw,
                zy2
            ))

    return slots
