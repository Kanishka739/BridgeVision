# 🌉 BridgeVision – Bridge Damage Detection

BridgeVision is a full-stack intelligent bridge damage detection system powered by YOLOv8 and Streamlit. Built for civil infrastructure monitoring, it enables offline analysis of uploaded bridge images using a custom-trained object detection model.

---

## 🚀 Features

- ⚙️ **YOLOv8-based Detection**: Custom-trained on 3,000+ bridge images annotated for cracks, corrosion, spalling, and leakage.
- 🖼️ **Streamlit UI**: Upload images and view predictions with bounding boxes and class names.
- 💡 **Offline Ready**: No internet required for inference after setup.
- 🔍 **Lightweight Full-Stack**: Combines deep learning backend with interactive frontend.

---

## 🧠 Model

- **Architecture**: YOLOv8 (Ultralytics)
- **Dataset**: ~3132 images 
- **Classes**: Crack, Corrosion, Spalling, Freelime,Leakage
- **Training**: Trained locally using VS Code

---

## 🛠️ Tech Stack

| Category         | Tools & Libraries                     |
|------------------|----------------------------------------|
| Model Training   | YOLOv8(Ultralytics) |
| Frontend         | Streamlit, PIL, OpenCV                 |
| Backend          | Python, NumPy, Torch                   |
| Deployment       | Local system (offline-ready)           |

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/BridgeVision.git
cd BridgeVision

## **Install dependencies**
pip install -r requirements.txt
