import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import SignLanguageModel
import time

# Cihaz ayarla (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Modeli yükle
model = SignLanguageModel().to(device)
model.load_state_dict(torch.load("model_epoch_50.pth", map_location=device))
model.eval()

# Görüntü işleme dönüşümleri
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Sınıf isimleri
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Kamerayı başlat
cap = cv2.VideoCapture(0)
fps_start = time.time()
frame_count = 0
fps = 0  # FPS başlangıçta tanımlandı ✅

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    h, w, _ = frame.shape

    # 📌 ROI Alanı (Dinamik Kare)
    roi_size = min(h, w) // 3
    x1, y1 = (w // 2 - roi_size // 2, h // 2 - roi_size // 2)
    x2, y2 = (x1 + roi_size, y1 + roi_size)

    roi = frame[y1:y2, x1:x2]

    image = transform(roi).unsqueeze(0).to(device)

    # 🔥 Model Tahmini
    with torch.no_grad():
        output = model(image)
        confidence, predicted = torch.max(output, 1)
        confidence_score = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
        letter = classes[predicted.item()]

    # 🌟 UI İyileştirmeleri
    overlay = frame.copy()
    alpha = 0.5
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 📌 Arka plan kutusu ekleyerek yazıyı belirginleştirme
    cv2.rectangle(frame, (30, 30), (300, 110), (0, 0, 0), -1)  # Siyah kutu (arka plan)
    cv2.putText(frame, f"Tahmin: {letter} ({confidence_score*100:.1f}%)", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)  # Beyaz yazı

    # ✅ FPS Hesaplama
    if frame_count >= 10:
        fps = frame_count / (time.time() - fps_start)
    
    # FPS göstergesi
    cv2.rectangle(frame, (30, 110), (250, 160), (0, 0, 0), -1)  # Siyah kutu
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
