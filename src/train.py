import torch
import torch.optim as optim
import torch.nn as nn
import time
from model import CNNModel
from dataset_loader import load_data
from dataset_loader import SignLanguageDataset
from torch.utils.data import DataLoader
import os

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Cihazı belirle (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Veriyi yükle
print("Veri yükleniyor...")  
train_loader = load_data(batch_size=16)
print(f"Veri başarıyla yüklendi! Toplam eğitim verisi: {len(train_loader.dataset)}")

# Modeli oluştur ve varsa önceden eğitilmiş modeli yükle
print("Model oluşturuluyor...")
model = CNNModel().to(device)

try:
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    print("Önceden eğitilmiş model yüklendi. Eğitime kaldığı yerden devam edilecek!")
except FileNotFoundError:
    print("Önceden eğitilmiş model bulunamadı. Model sıfırdan eğitilecek.")

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Kaç epoch boyunca eğitilecek?
num_epochs = 100  
save_interval = 25  

# Log dosyası aç
log_file = open("training_log.txt", "a")  

start_time = time.time()

# Eğitim döngüsü
print("Eğitime başlanıyor...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} başlıyor...")  
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    log_text = f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f} sec\n"

    print(log_text, end="")  
    log_file.write(log_text)  

    # Modeli kaydet (her 10 epoch'ta bir)
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

# Son modeli kaydet
torch.save(model.state_dict(), "model_final.pth")
print("Final model saved!")

# 🚀 TEST DOĞRULUK HESAPLAMA 🚀
print("\n📌 Model test ediliyor...")

# Test veri setini yükle
test_data_path = "dataset/tr_signLanguage_dataset/test"

if not os.path.exists(test_data_path):
    print(f"❌ HATA: Test veri klasörü bulunamadı! Lütfen '{test_data_path}' klasörünü kontrol et.")
    exit()

test_dataset = SignLanguageDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test verisi başarıyla yüklendi! Toplam test verisi: {len(test_dataset)}")

# İlk birkaç test verisinin etiketini gösterelim (doğru veri yüklendi mi kontrol için)
print("İlk 5 test verisinin etiketleri:", [test_dataset.labels[i] for i in range(min(5, len(test_dataset)))])

model.eval()  
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n✅ 📌 Test Doğruluk Oranı: {accuracy:.2f}%")

# Log dosyasına da ekleyelim
log_file.write(f"\n📌 Test Doğruluk Oranı: {accuracy:.2f}%\n")

log_file.close()
