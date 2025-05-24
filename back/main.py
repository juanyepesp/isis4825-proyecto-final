from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import uvicorn
from models import *
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    contents = await file.read()
    input_img = Image.open(io.BytesIO(contents))
    
    if model_name == "YOLO_Small":
        pred_img = predict_yolo(img=input_img, model=model_name)
    elif model_name == "YOLO_Large":
        pred_img = predict_yolo(img=input_img, model=model_name)
    elif model_name == "YOLO_Large_Congelando":
        pred_img = predict_yolo(img=input_img, model=model_name)
    elif model_name == "Faster_RCNN":
        pred_img = predict_faster_rcnn(img=input_img)
    elif model_name == "RetinaNet":
        pred_img = predict_retinanet(img=input_img)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")