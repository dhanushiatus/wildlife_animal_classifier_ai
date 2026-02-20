## 🌍 Wildlife Animal Species Classifier

This project is an AI-powered wildlife recognition system capable of identifying **90 different animal species** from images using deep learning and computer vision techniques.

The system combines:

- **EfficientNetB0** (pretrained on ImageNet) for high-accuracy feature extraction
- **YOLOv8** for object detection (if enabled)
- **OpenCV** for image preprocessing and handling
- **Streamlit** for interactive web deployment

### 🧠 Technical Workflow

1. Image input is processed using **OpenCV**
2. (Optional) **YOLOv8** detects and isolates the animal in the image
3. The cropped image is resized to 224x224
4. Features are extracted using **EfficientNetB0**
5. A custom-trained dense layer performs 90-class softmax classification
6. Results are displayed with confidence scores in real-time

### 🚀 Key Features

- 90 Animal Species Classification
- High validation accuracy (99%+ during training)
- Real-time prediction system
- Top-5 probability predictions
- Transfer Learning-based architecture
- Lightweight and deployable Streamlit app

### 🔬 Technologies Used

- TensorFlow / Keras
- EfficientNetB0
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Streamlit
