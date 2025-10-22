from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from image_predict import ImagePredict
from config_loader import config_learn

app = FastAPI()

config_path='config.yaml'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/recognize/{filename}")
def recognize_image(filename: str):
    model_path = config_learn['save_model_path']
    image_path = os.path.join("uploads", filename)
    if not os.path.exists(image_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    img_pred = ImagePredict()
    img_pred.readModel(model_path)
    img_pred.readImage(image_path)
    predicted_class, confidence = img_pred.predict()
    return {
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2)  # パーセンテージ表示
    }

