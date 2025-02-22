from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import shutil
from pathlib import Path
from fastapi.responses import FileResponse
import uvicorn
from pdf2image import convert_from_path
import imghdr

def convert_pdf_to_image(pdf_path: str):
    images = convert_from_path(pdf_path)
    if images:
        image_path = pdf_path.replace(".pdf", ".png")
        images[0].save(image_path, "PNG")
        return image_path
    raise HTTPException(status_code=400, detail="Ошибка: не удалось конвертировать PDF в изображение.")

def rotate_and_align(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = [line[0][1] for line in lines]
        median_angle = np.median(angles)
        angle_degrees = np.rad2deg(median_angle) - 90
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

def split_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Ошибка: не удалось загрузить изображение {image_path}.")
    h, w = image.shape[:2]
    middle = w // 2
    left_half = image[:, :middle]
    right_half = image[:, middle:]
    left_path = image_path.replace(".png", "_left.png").replace(".jpg", "_left.jpg")
    right_path = image_path.replace(".png", "_right.png").replace(".jpg", "_right.jpg")
    cv2.imwrite(left_path, left_half)
    cv2.imwrite(right_path, right_half)
    return left_path, right_path

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def process_images(image1_path: str, image2_path: str):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        raise HTTPException(status_code=400, detail=f"Ошибка: не удалось загрузить одно из изображений: {image1_path}, {image2_path}.")
    img1 = rotate_and_align(img1)
    img2 = rotate_and_align(img2)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    _, diff_thresh = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
    overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    output_path = UPLOAD_DIR / "comparison_result.png"
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(None)):
    file1_path = UPLOAD_DIR / file1.filename
    with file1_path.open("wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    
    file1_format = imghdr.what(str(file1_path))
    if file1.filename.lower().endswith(".pdf"):
        file1_path = Path(convert_pdf_to_image(str(file1_path)))
    elif file1_format not in ["jpeg", "png"]:
        raise HTTPException(status_code=400, detail=f"Ошибка: неподдерживаемый формат файла {file1.filename}. Ожидается PNG или JPG, но получен {file1_format}.")
    
    if file2:
        file2_path = UPLOAD_DIR / file2.filename
        with file2_path.open("wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)
        file2_format = imghdr.what(str(file2_path))
        if file2.filename.lower().endswith(".pdf"):
            file2_path = Path(convert_pdf_to_image(str(file2_path)))
        elif file2_format not in ["jpeg", "png"]:
            raise HTTPException(status_code=400, detail=f"Ошибка: неподдерживаемый формат файла {file2.filename}. Ожидается PNG или JPG, но получен {file2_format}.")
    else:
        left_path, right_path = split_image(str(file1_path))
        file1_path, file2_path = left_path, right_path
    
    result_path = process_images(str(file1_path), str(file2_path))
    return FileResponse(result_path, media_type="image/png", filename="comparison_result.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
