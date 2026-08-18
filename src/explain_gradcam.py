import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from train import MultiModalClassifier


device = torch.device("cpu")
model = MultiModalClassifier(num_sensor_features=4).to(device)
model.load_state_dict(torch.load('models/multimodal_model.pth', map_location=device))
model.eval() 


from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
train_dir = base_dir / "Data" / "row" / "images" / "train"


image_files = list(train_dir.glob("**/*.[jJ][pP]*[gG]")) + list(train_dir.glob("**/*.[pP][nN][gG]"))

if not image_files:
    raise FileNotFoundError(f"there are not any photos found in folder: {train_dir}")

image_path = str(image_files[0])
print(f"Loading image from: {image_path}")

image_path = image_files[0]
print(f"Loading image from: {image_path}") 
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] 
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img_float = np.float32(rgb_img) / 255 


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0).to(device)


sensor_tensor = torch.tensor([[85.0, 4.5, 95.0, 0.8]], dtype=torch.float32).to(device)


target_layers = [model.vision_backbone.features[-1]]


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, sensors):
        super().__init__()
        self.model = model
        self.sensors = sensors
    def forward(self, x):
        return self.model(x, self.sensors)

wrapped_model = ModelWrapper(model, sensor_tensor)

cam = GradCAM(model=wrapped_model, target_layers=target_layers)


grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title('Original Defect Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title('Grad-CAM Heatmap (Explainability)')
plt.axis('off')

plt.tight_layout()
plt.savefig('models/gradcam_result.png')
print("Grad-CAM visualization saved to models/gradcam_result.png")
plt.show()