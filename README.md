# 🛡️ Multi-Modal Surface Defect Inspection System

An end-to-end industrial quality control system combining **Computer Vision (CNN)** and **Sensor Telemetry (Temperature, Vibration, Pressure, Humidity)** using **Early Fusion** for automated defect detection.

![Streamlit Dashboard Demo](https://via<img width="1920" height="1080" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/7853ad2c-f57f-4c2f-95a8-bc6e1cc39103" />
.placeholder.com/800x400.png?text=Add+Your+Dashboard+Screenshot+Here)

---

## 🌟 Key Features

* **Multi-Modal Early Fusion:** Combines spatial image features with physical numerical sensor readings.
* **Explainable AI (XAI):** Integrated **Grad-CAM** heatmaps to visualize exact regions driving defect classifications.
* **Interactive Dashboard:** Built with **Streamlit** allowing real-time image uploads and dynamic sensor adjustments.
* **Model Optimization:** Trained in PyTorch and exported to **ONNX Runtime** for high-efficiency inference.

---

## 🏗️ System Architecture

1. **Vision Feature Extractor:** CNN Backbone processing surface images ($224 \times 224$).
2. **Sensor Processing Stream:** Multi-Layer Perceptron (MLP) normalizing numerical telemetry data.
3. **Early Fusion Layer:** Concatenates visual vector features with numerical sensor embeddings before final classification logits.

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Multi-Modal-Defect-Inspection.git](https://github.com/YOUR_USERNAME/Multi-Modal-Defect-Inspection.git)
   cd Multi-Modal-Defect-Inspection
