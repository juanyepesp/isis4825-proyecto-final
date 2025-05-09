from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from filter import process_image_array


model = YOLO("models/best.pt")

def predict_yolo(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert("RGB")) 
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    preprocessed = process_image_array(img_bgr)
    results = model(preprocessed)
    annotated = results[0].plot()
    return Image.fromarray(annotated)
