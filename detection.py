from ultralytics import YOLO

# Load model
model = YOLO(r"E:\object dectection\runs\detect\training\weights\best.pt")

# Run live detection from webcam (0 = default camera)
model.predict(
    source=0,
    conf=0.5,
    show=True
)
