from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import shutil
from pathlib import Path
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def process_images(image1_path: str, image2_path: str):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    diff = cv2.absdiff(img1, img2)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    output_path = UPLOAD_DIR / "comparison_result.png"
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    file1_path = UPLOAD_DIR / file1.filename
    file2_path = UPLOAD_DIR / file2.filename
    
    with file1_path.open("wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with file2_path.open("wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)
    
    result_path = process_images(str(file1_path), str(file2_path))
    return FileResponse(result_path, media_type="image/png", filename="comparison_result.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
