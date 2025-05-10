import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel
from dataset_loader import SignLanguageDataset

# Cihazı belirle (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Modeli yükle
model = CNNModel().to(device)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.eval()

# Test verisini yükle
test_dataset = SignLanguageDataset("dataset/tr_signLanguage_dataset/test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Doğruluk hesaplama
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Sonucu ekrana yazdır
accuracy = 100 * correct / total
print(f"\n📌 Test Doğruluk Oranı: {accuracy:.2f}%")
