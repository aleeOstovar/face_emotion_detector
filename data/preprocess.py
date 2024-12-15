import cv2
import numpy as np
from albumentations import Compose, RandomCrop, HorizontalFlip, Normalize, Resize
from albumentations.core.composition import OneOf
from albumentations.augmentations.transforms import RandomBrightnessContrast, GaussianBlur

class ImagePreprocessor:
    def __init__(self, image_size=(48, 48), augment=False):
        """
        Initializes the ImagePreprocessor.

        Args:
            image_size (tuple): Target size for resizing images (height, width).
            augment (bool): Whether to apply data augmentation during preprocessing.
        """
        self.image_size = image_size
        self.augment = augment
        self.augmentation_pipeline = self._build_augmentation_pipeline()

    def _build_augmentation_pipeline(self):
        """
        Creates a data augmentation pipeline using Albumentations.

        Returns:
            Compose: Albumentations composition of augmentations.
        """
        return Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.5),
            OneOf([
                GaussianBlur(blur_limit=3, p=0.5),
                RandomCrop(height=self.image_size[0] - 4, width=self.image_size[1] - 4, p=0.5)
            ], p=0.7),
            Resize(height=self.image_size[0], width=self.image_size[1]),
            Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0)
        ])

    def preprocess_image(self, image):
        """
        Preprocesses a single image: resizing, normalizing, and optionally augmenting.

        Args:
            image (numpy.ndarray): Input image to preprocess.

        Returns:
            numpy.ndarray: Preprocessed image.
        """
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Ensure grayscale images have a channel dimension

        image = cv2.resize(image, self.image_size)  # Resize to target dimensions

        if self.augment:
            augmented = self.augmentation_pipeline(image=image)
            image = augmented['image']

        # Normalize manually if not using Albumentations
        image = (image / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

        return np.expand_dims(image, axis=0)  # Add batch dimension

    def preprocess_batch(self, images):
        """
        Preprocesses a batch of images.

        Args:
            images (list of numpy.ndarray): List of images to preprocess.

        Returns:
            numpy.ndarray: Batch of preprocessed images.
        """
        return np.array([self.preprocess_image(image) for image in images])

# Example Usage
if __name__ == "__main__":
    # Mock data: an example grayscale image
    mock_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    # Initialize preprocessor with augmentation enabled
    preprocessor = ImagePreprocessor(image_size=(48, 48), augment=True)

    # Preprocess a single image
    preprocessed_image = preprocessor.preprocess_image(mock_image)
    print("Preprocessed image shape:", preprocessed_image.shape)
