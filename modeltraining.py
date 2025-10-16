from ultralytics import YOLO

# 1️⃣ Load a pre-trained YOLOv8 model (you can change the variant)
model = YOLO("yolov8n.pt")  # yolov8n.pt = Nano, fast & light
# You can also use: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 2️⃣ Train the model
results = model.train(
    data=r"Dataset/data.yaml",  # path to your dataset YAML file
    epochs=100,                # number of training epochs
    imgsz=640,                 # image size (default: 640)
    batch=16,                  # adjust based on GPU memory
    device='cpu',                  # set to 0 for GPU, or 'cpu' for CPU
    name="training" # experiment name (saved under runs/train/)
)

# 3️⃣ Evaluate the model after training
metrics = model.val()  # runs validation using the best.pt weights

