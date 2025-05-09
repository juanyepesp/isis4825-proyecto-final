from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import uvicorn
from yolo import predict_yolo

app = FastAPI()

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

    if model_name == "modelo_1":
        pred_img = input_img.convert("L")  # Grayscale
    elif model_name == "modelo_2":
        sepia_filter = Image.new("RGB", input_img.size)
        sepia_data = [
            (int(r * 0.393 + g * 0.769 + b * 0.189),
             int(r * 0.349 + g * 0.686 + b * 0.168),
             int(r * 0.272 + g * 0.534 + b * 0.131))
            for r, g, b in input_img.getdata()
        ]
        sepia_filter.putdata(sepia_data)
        pred_img = sepia_filter
    elif model_name == "modelo_3":
        pred_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
        
    elif model_name == 'yolo_v11s':
        pred_img = predict_yolo(img=input_img)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
