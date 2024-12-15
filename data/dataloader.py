import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import preprocess

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to each image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        """Helper function to load all image paths from the dataset directory."""
        return [
            os.path.join(self.data_dir, fname)
            for fname in os.listdir(self.data_dir)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and apply transformations."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def create_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a PyTorch DataLoader.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        num_workers (int, optional): Number of worker processes for data loading. Default is 4.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    transform = preprocess.get_preprocessing_pipeline()

    dataset = CustomDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader

# Example Usage
if __name__ == "__main__":
    DATA_DIR = "path/to/your/dataset"
    BATCH_SIZE = 16

    dataloader = create_dataloader(DATA_DIR, batch_size=BATCH_SIZE)

    for batch_idx, images in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {images.shape}")
