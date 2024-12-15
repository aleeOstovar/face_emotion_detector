# data/data_loader.py
import os
import torch
from torchvision import datasets, transforms

class FERDataLoader:
    def __init__(self, data_dir, batch_size, img_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def load_data(self):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')

        train_data = datasets.ImageFolder(train_dir, transform=self.transforms)
        val_data = datasets.ImageFolder(val_dir, transform=self.transforms)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
