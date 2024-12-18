from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import uuid
import os
from PIL import Image
from ultralytics import YOLO
import easyocr

app = FastAPI()

# Model Initialization
COCO_MODEL_DIR = "./models/yolov8n.pt"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
FOLDER_PATH = "./licenses_plates_imgs_detected/"

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
reader = easyocr.Reader(['en'], gpu=False)


# Utility Function for OCR
def read_license_plate(license_plate_crop):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if not detections:
        return None, 0

    plate = []
    for result in detections:
        text, score = result[1], result[2]
        text = text.upper()
        scores += score
        plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    return None, 0


# Model Prediction
def model_prediction(image):
    results = []
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    license_detections = license_plate_detector(img)[0]

    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            text, text_score = read_license_plate(license_plate_crop_gray)

            if text:
                # Save Cropped Image
                img_name = f"{uuid.uuid4()}.jpg"
                img_path = os.path.join(FOLDER_PATH, img_name)
                cv2.imwrite(img_path, license_plate_crop)

                results.append({
                    "text": text,
                    "score": text_score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "image_url": img_path
                })

    return results


# API Endpoint
@app.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        detections = model_prediction(image)

        if not detections:
            return {"message": "No license plates detected"}
        
        # Extract and clean plate text
        plate_texts = [detection["text"].replace(" ", "") for detection in detections]

        # Join results into one line
        return {"plates": " ".join(plate_texts)}

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
