
---

# 🧠 YOLOv8 Object Detection Project

## 📌 Project Overview

This project demonstrates how to train and use a **YOLOv8 (You Only Look Once v8)** model for custom object detection using the **Ultralytics YOLO** library.

The model is trained on a **custom dataset** and can perform **real-time object detection** on:

* Images 🖼️
* Videos 🎥
* Webcam streams 📷

YOLOv8 is a state-of-the-art, one-stage object detection architecture that provides high accuracy and real-time performance for edge and embedded devices.

---

## 🧱 Folder Structure

Ensure your dataset is organized as follows before training:

If you’re not familiar with how to create your own dataset — including image labeling and annotation generation — you can refer to the following guide:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```


## 💻 Choose a Code Editor

You can use any IDE or code editor such as:

* **Visual Studio Code (VS Code)**
* **Eclipse**
* **PyCharm**

---

## 🧩 Setting Up Your Environment

### 1️⃣ Create and Activate a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv
```

#### For Windows:

```bash
venv\Scripts\activate
```

#### For Linux/Mac:

```bash
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

Make sure your virtual environment is active, then install the required libraries:

```bash
pip install ultralytics torch torchvision opencv-python pyyaml
```

Alternatively, create a `requirements.txt` file and run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

You can train your YOLOv8 model using the `modeltraining.py` script or directly from the terminal.

### Option 1: Using Python Script

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (e.g., small version)
model = YOLO("yolov8s.pt")

# Train the model on your custom dataset
model.train(data="dataset/data.yaml", epochs=100, imgsz=640, batch=16, name="custom_yolo_train")
```

### Option 2: Using Command Line

```bash
yolo detect train data=dataset/data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16 name=custom_yolo_train
```

---

## 🏋️‍♀️ Training Output

For this project, I trained the YOLOv8 model to detect three custom object classes.
Below are the results obtained after training, and I’ve also included my training outputs and evaluation results under the runs/ directory.

![val_batch0_pred](https://github.com/user-attachments/assets/4e6add1c-4888-454b-8db2-bb4d4d6e1356)



After training, YOLOv8 will automatically create a directory:

```
runs/train/custom_yolo_train/
```

Inside it, you’ll find:

* `weights/best.pt` → **Best performing model**
* `weights/last.pt` → Model state from last epoch
* `results.csv` → Training logs
* `results.png` → Training graphs (loss, mAP, precision, recall)

---

## 🧠 Running Inference

Once you have your trained weights (`best.pt`), use the following script to detect objects on images, videos, or webcam:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/custom_yolo_train/weights/best.pt")

# Run inference on an image
results = model.predict(source="test_images/image1.jpg", conf=0.5, show=True)
```

### Other Options:

* **Images:**
  `source="path/to/image.jpg"`
* **Videos:**
  `source="path/to/video.mp4"`
* **Webcam:**
  `source=0`

Detected results will be saved in:

```
runs/detect/predict/
```

---

## 📊 Results & Evaluation

After training, YOLOv8 automatically computes performance metrics such as:

* **mAP (Mean Average Precision)**
* **Precision**
* **Recall**
* **F1 Score**

You can visualize these metrics and training curves inside:

```
runs/train/custom_yolo_train/
```

---

## 📁 Model Weights

Best weights will be saved at:

```
runs/train/custom_yolo_train/weights/best.pt
```

Use this file for inference, evaluation, or deployment.

---

## 🧾 Optional: Export the Model

YOLOv8 supports exporting to multiple formats for deployment:

```bash
yolo export model=runs/train/custom_yolo_train/weights/best.pt format=onnx
```

Available formats: `onnx`, `torchscript`, `coreml`, `tflite`, `openvino`, `engine` (TensorRT).

---

## 🛠️ Troubleshooting

| Problem                 | Possible Cause                                  | Solution                                                |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| **No labels found**     | Label files missing or not matching image names | Ensure each image has a `.txt` label with the same name |
| **CUDA not available**  | PyTorch not installed with GPU support          | Reinstall PyTorch with CUDA enabled                     |
| **Low mAP / accuracy**  | Poor labeling or small dataset                  | Increase dataset size or improve label quality          |
| **Out of memory (OOM)** | Batch size too large for GPU                    | Reduce batch size or use smaller image size             |

---

## 📚 References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [YOLO Paper Series (v1–v8)](https://pjreddie.com/darknet/yolo/)
* [PyTorch Official Docs](https://pytorch.org/docs/)

---



  **SAI CHAITANYA KANCHARANA**
📧kancharanasaichaitanya@gmail.com
🌐https://github.com/saichaitanya-c/

---

