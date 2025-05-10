import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))  # Örn: ["A", "B", ..., "Z", "space", "del", "nothing"]
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),     # Veri artırma
            transforms.RandomRotation(15),         # Hafif döndürme
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def load_data(batch_size=32):
    train_dataset = SignLanguageDataset("dataset/tr_signLanguage_dataset/train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
