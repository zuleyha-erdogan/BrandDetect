# 🚗 Car Brand Detection Using YOLOv8 and Roboflow

This project uses the **YOLOv8** object detection model to identify car brands in images. The dataset is managed and downloaded via **Roboflow**, and training is performed in Python (e.g., Google Colab).


## 🧰 Technologies and Tools

- Python
- Roboflow (Dataset management & download)
- Ultralytics YOLOv8 (object detection)
- OpenCV (image processing)
- Matplotlib (visualization)
- Google Colab (optional, for easy setup)

## ⚙️ Setup & Installation

Install required packages:

```bash
!pip install roboflow
!pip install ultralytics
from roboflow import Roboflow

📥 Dataset Download (Roboflow)
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE_NAME").project("YOUR_PROJECT_NAME")
version = project.version(1)
dataset = version.download("yolov8")

🚀 Model Training

from ultralytics import YOLO
Train YOLOv8 model with your dataset:


model = YOLO("yolov8n.pt")  # Use YOLOv8 nano pre-trained weights

model.train(
    data="/content/Car-Brand-Logos-1/data.yaml",  # Path to your dataset config
    epochs=10,
    imgsz=640,
    batch=16,
    project="car_brand_detect",
    name="yolov8n_v1",
    val=True
)
🎯 Model Inference (Prediction)
Load the trained model and run prediction on an image:

python
Kopyala
Düzenle
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("car_brand_detect/yolov8n_v14/weights/best.pt")

image_path = "/content/resim.jpg"
results = model.predict(image_path, show=False)

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for result in results:
    plotted_img = result.plot()
    plt.figure(figsize=(10, 8))
    plt.imshow(plotted_img)
    plt.axis("off")
    plt.title("Car Brand Detection Result")
    plt.show()

📁 File Upload (Google Colab)
If you want to upload images manually in Colab:

from google.colab import files
uploaded = files.upload()

⚠️ Notes
Ensure the dataset path (data.yaml) is correct.

Replace API keys and workspace/project names with your own Roboflow details.

Adjust training parameters like epochs, batch size, and imgsz as needed.

YOLOv8 nano (yolov8n.pt) is a lightweight model; for better accuracy try larger versions.


![brand](https://github.com/user-attachments/assets/b58e83d5-7940-4daf-82c6-1c9842de16d3)







