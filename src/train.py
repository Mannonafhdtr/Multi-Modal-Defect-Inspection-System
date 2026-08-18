import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import MultiModalDefectDataset

# Define Multi-Modal Architecture (Early Fusion of Vision & Tabular)
class MultiModalClassifier(nn.Module):
    def __init__(self, num_sensor_features=4):
        super(MultiModalClassifier, self).__init__()
        
        # Backbone vision model (MobileNetV3 Small)
        self.vision_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_ftrs = self.vision_backbone.classifier[0].in_features
        self.vision_backbone.classifier = nn.Identity()  # Remove final classification layer

        # Dense network for tabular sensor features
        self.sensor_fc = nn.Sequential(
            nn.Linear(num_sensor_features, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary output: Normal vs Defective
        )

    def forward(self, image, sensors):
        img_features = self.vision_backbone(image)
        sensor_features = self.sensor_fc(sensors)
        
        # Concatenate vision and tabular representation vectors
        fused_features = torch.cat((img_features, sensor_features), dim=1)
        output = self.classifier(fused_features)
        return output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    csv_path = os.path.join('Data', 'row', 'processed', 'sensors_data.csv')
    
    train_dataset = MultiModalDefectDataset(csv_file=csv_path, split='train', transform=data_transforms['train'])
    val_dataset = MultiModalDefectDataset(csv_file=csv_path, split='validation', transform=data_transforms['validation'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = MultiModalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simple training loop for 3 epochs
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            sensors = batch['sensors'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, sensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Save PyTorch Model
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)
    pytorch_model_path = os.path.join(models_dir, 'multimodal_model.pth')
    torch.save(model.state_dict(), pytorch_model_path)
    print(f"PyTorch model saved to {pytorch_model_path}")

    # Convert and Export to ONNX Format
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_sensors = torch.randn(1, 4).to(device)
    onnx_path = os.path.join(models_dir, 'multimodal_model.onnx')

    torch.onnx.export(
        model,
        (dummy_image, dummy_sensors),
        onnx_path,
        input_names=['image', 'sensors'],
        output_names=['output'],
        dynamic_axes={'image': {0: 'batch_size'}, 'sensors': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Model successfully optimized and exported to ONNX format at {onnx_path}!")

if __name__ == '__main__':
    main()