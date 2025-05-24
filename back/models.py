from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from filter import process_image_array
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn


def predict_yolo(img: Image.Image, model:str) -> Image.Image:
    if model is None:
        raise ValueError("Model is not loaded. Please load a YOLO model before prediction.")
    elif model not in ["YOLO_Small", "YOLO_Large", "YOLO_Large_Congelando", "Faster_RCNN", "RetinaNet"]:
        raise ValueError(f"Unsupported model: {model}")
    elif model == "YOLO_Small":
        model = YOLO("models/model1.pt")
    elif model == "YOLO_Large":
        model = YOLO("models/model2.pt")
    elif model == "YOLO_Large_Congelando":
        model = YOLO("models/model3.pt")
    
    img_np = np.array(img.convert("RGB")) 
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    preprocessed = process_image_array(img_bgr)
    results = model.predict(preprocessed)
    annotated = results[0].plot()
    return Image.fromarray(annotated)


def predict_faster_rcnn(img: Image.Image) -> Image.Image:
    model_path = "models/faster_rcnn_best.pth"
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    img_np = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    preprocessed = process_image_array(img_bgr)
    preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(preprocessed_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    for i, (box, score) in enumerate(zip(prediction['boxes'], prediction['scores'])):
        if score > 0.5:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(preprocessed_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = prediction['labels'][i]
            class_name = "rio" if label == 2 else "carretera"
            cv2.putText(preprocessed_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return Image.fromarray(preprocessed_rgb)


def predict_retinanet(img: Image.Image) -> Image.Image:
    model_path = "models/retinanet_best.pth"
    model = retinanet_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    img_np = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    preprocessed = process_image_array(img_bgr)
    preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(preprocessed_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    for i, (box, score) in enumerate(zip(prediction['boxes'], prediction['scores'])):
        if score > 0.3:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(preprocessed_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = prediction['labels'][i]
            class_name = "rio" if label == 1 else "carretera"
            cv2.putText(preprocessed_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return Image.fromarray(preprocessed_rgb)