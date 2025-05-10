from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from src.model import SignLanguageModel
from src.cuda import get_device
import numpy as np
import logging
import time

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model yükleme
try:
    device = get_device()
    model = SignLanguageModel()
    model.load_state_dict(torch.load('model_epoch_25.pth', map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
    raise

def process_frame(frame):
    try:
        # Görüntüyü işle
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (64, 64))
        frame_tensor = torch.from_numpy(frame_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        frame_tensor = frame_tensor.to(device)

        # Tahmin yap
        with torch.no_grad():
            output = model(frame_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Tahmin sonucunu görüntüye ekle
        label = chr(prediction + ord('A'))
        confidence_text = f"{confidence:.2%}"
        cv2.putText(frame, f"Tahmin: {label} ({confidence_text})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, label, confidence
    except Exception as e:
        logger.error(f"Frame işlenirken hata oluştu: {str(e)}")
        return frame, "Hata", 0.0

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Kamera açılamadı!")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Frame okunamadı")
                break

            frame, label, confidence = process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("Frame encode edilemedi")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logger.error(f"Video akışı sırasında hata: {str(e)}")
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        'status': 'active',
        'device': str(device),
        'model_loaded': True
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 