import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from train import MultiModalClassifier

# page charactaristics
st.set_page_config(page_title="Multi-Modal Defect Inspection", layout="wide")

st.title("🛡️ Multi-Modal Surface Defect Inspection System")
st.markdown("Early Fusion AI for Automated Defect Detection using Computer Vision and Sensor Telemetry.")

# One Time Model DownLoading
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = MultiModalClassifier(num_sensor_features=4).to(device)
    model_path = Path(__file__).resolve().parent.parent / "models" / "multimodal_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Sidebar for Sensors Reading
st.sidebar.header(" Sensor Telemetry Inputs")
temp = st.sidebar.slider("Temperature (°C)", 20.0, 120.0, 75.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 10.0, 2.5)
pressure = st.sidebar.slider("Pressure (PSI)", 50.0, 150.0, 100.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 1.0, 0.4)

# Uploading the Photo
uploaded_file = st.file_uploader("Upload Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # The Photo Processing
    pil_img = Image.open(uploaded_file).convert('RGB')
    rgb_img = np.array(pil_img)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img_resized).unsqueeze(0).to(device)
    sensor_tensor = torch.tensor([[temp, vibration, pressure, humidity]], dtype=torch.float32).to(device)

    # the prediction
    with torch.no_grad():
        output = model(input_tensor, sensor_tensor)

    # results processing whether its classes or 2 binary classification
    if output.shape[-1] == 1:
        prob = torch.sigmoid(output).item()
        is_defective = prob > 0.5
        confidence = prob if is_defective else (1 - prob)
    else:
        probs = torch.softmax(output, dim=1).squeeze(0)
        # Assuming class 1 corresponds to 'Defective'
        defective_prob = probs[1].item() if len(probs) > 1 else probs[0].item()
        is_defective = torch.argmax(probs).item() == 1
        confidence = defective_prob if is_defective else (1 - defective_prob)

    # showing results
    st.subheader("Inspection Verdict")
    col_res, col_conf = st.columns(2)
    
    if is_defective:
        col_res.error("❌ DEFECT DETECTED")
    else:
        col_res.success("✅ NO DEFECT DETECTED")
        
    col_conf.metric("Confidence Score", f"{confidence * 100:.2f}%")

    # Grad-CAM Explainability
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

    # Showing Photo
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_img_resized, caption="Uploaded Surface Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM Heatmap (Explainability)", use_container_width=True)