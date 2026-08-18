import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MultiModalDefectDataset(Dataset):
    """
    Custom Dataset to load image and sensor data simultaneously.
    """
    def __init__(self, csv_file, split='train', transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            split (string): Dataset split ('train' or 'validation').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform

        # Sensor feature columns
        self.sensor_cols = ['temperature_c', 'vibration_mms', 'humidity_pct', 'line_speed_ms']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        # Extract sensor tabular features
        sensor_features = self.df.iloc[idx][self.sensor_cols].values.astype('float32')
        sensor_tensor = torch.tensor(sensor_features, dtype=torch.float32)

        # Target label (binary defect label)
        label = self.df.iloc[idx]['has_defect']
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image,
            'sensors': sensor_tensor,
            'label': label_tensor
        }