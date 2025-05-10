import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Veri seti yolları
train_dir = "dataset/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 64  # Görselleri 64x64'e ölçekleyelim

# Etiketleri almak için klasör isimlerini oku
classes = sorted(os.listdir(train_dir))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Siyah beyaza çevir
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Yeniden boyutlandır
    transforms.ToTensor(),  # Tensor formata çevir
    transforms.Normalize((0.5,), (0.5,))  # Normalize et (-1,1 aralığı)
])

# Özel PyTorch veri kümesi sınıfı
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Verileri oku
        for label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, label)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append((img_path, class_to_idx[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")  # Gri tonlama
        if self.transform:
            image = self.transform(image)
        return image, label

# Veri setini yükle
dataset = ASLDataset(train_dir, transform=transform)

# Eğitim ve doğrulama veri kümesi olarak ayır
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Veri yükleyicileri oluştur
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Kontrol için veri seti boyutları
print(f"Toplam veri: {len(dataset)}")
print(f"Eğitim verisi: {len(train_dataset)}")
print(f"Doğrulama verisi: {len(val_dataset)}")
