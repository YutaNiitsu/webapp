from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from image_predict import ImagePredict
from config_loader import config_learn
from pathlib import Path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 分析
@app.get("/recognize/{filename}")
def recognize_image(filename: str):
    output = config_learn.get('output')
    model_path = os.path.join('/mnt/c/Users/yniit/Documents/aitraining', output.get('save_model_path'))
    image_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(image_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    img_pred = ImagePredict()
    img_pred.readModel(model_path)
    img_pred.readImage(image_path)
    predicted_class, confidence = img_pred.predict()

    # 分析後にファイル削除
    try:
        os.remove(image_path)
    except Exception as e:
        print(f"[WARN] ファイル削除失敗: {e}")
    
    return {
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2)  # パーセンテージ表示
    }

# アップロード
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return JSONResponse(content={"filename": file.filename, "status": "uploaded"})