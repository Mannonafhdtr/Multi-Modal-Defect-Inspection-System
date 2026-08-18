import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths based on the directory structure
images_dir = os.path.join('Data', 'row', 'images')
output_dir = os.path.join('Data', 'row', 'processed')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Collect all image file paths along with their dataset split and category
records = []
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

for split in ['train', 'validation']:
    split_dir = os.path.join(images_dir, split)
    if os.path.exists(split_dir):
        for root, _, files in os.walk(split_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, f)
                    category = os.path.basename(root)  # Extract category/defect type
                    records.append({
                        'image_path': full_path,
                        'split': split,
                        'category': category
                    })

# Convert records to DataFrame
df_images = pd.DataFrame(records)
n_samples = len(df_images)

print(f"Found {n_samples} images across train and validation sets.")

# Generate synthetic sensor telemetry data
temperature = np.random.normal(loc=65.0, scale=10.0, size=n_samples)  # Temperature in Celsius
vibration = np.random.normal(loc=2.5, scale=0.8, size=n_samples)      # Vibration level in mm/s
humidity = np.random.normal(loc=45.0, scale=5.0, size=n_samples)       # Humidity percentage
line_speed = np.random.normal(loc=1.2, scale=0.2, size=n_samples)     # Production line speed in m/s

# Formulate domain logic for defect likelihood based on sensor anomalies
defect_score = (
    (temperature - 65.0) * 0.04 + 
    (vibration - 2.5) * 0.5 + 
    np.random.normal(0, 0.5, size=n_samples)
)

# Populate features and binary defect label (0: Normal, 1: Defective)
df_images['temperature_c'] = np.round(temperature, 2)
df_images['vibration_mms'] = np.round(vibration, 2)
df_images['humidity_pct'] = np.round(humidity, 2)
df_images['line_speed_ms'] = np.round(line_speed, 2)
df_images['has_defect'] = (defect_score > 0.5).astype(int)

# Export generated dataset to CSV
output_path = os.path.join(output_dir, 'sensors_data.csv')
df_images.to_csv(output_path, index=False)

print(f"Dataset successfully created and saved to: {output_path}")
print(df_images.head())