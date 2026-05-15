def detect_vehicles_yolo(frame, model):
    res = model.predict(
        frame,
        classes=[2, 3, 5, 7],
        conf=0.20,
        iou=0.40,
        verbose=False
    )

    boxes = []

    if res and res[0].boxes is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist()

    return boxes
