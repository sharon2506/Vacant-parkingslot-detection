@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8s.pt")
