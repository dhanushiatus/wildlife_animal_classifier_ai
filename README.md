# 🌍 Wildlife Animal Species Classifier

An AI-powered wildlife recognition system capable of identifying **90 different animal species** from images using deep learning and computer vision.

The system combines:

- **EfficientNetB0** (pretrained on ImageNet) for high-accuracy feature extraction  
- **YOLOv8** (optional) for object detection  
- **OpenCV** for image preprocessing and handling  
- **Streamlit** for interactive web deployment  

---

## 🧠 Technical Workflow

1. Image input is processed using **OpenCV**  
2. (Optional) **YOLOv8** detects and isolates the animal in the image  
3. The cropped image is resized to 224x224  
4. Features are extracted using **EfficientNetB0**  
5. A custom-trained dense layer performs 90-class softmax classification  
6. Results are displayed with confidence scores in real-time  

---

## 🚀 Key Features

- 90 Animal Species Classification  
- High validation accuracy (99%+ during training)  
- Real-time prediction system  
- Top-5 probability predictions  
- Transfer Learning-based architecture  
- Lightweight and deployable Streamlit app  

---

## 🔬 Technologies Used

- TensorFlow / Keras  
- EfficientNetB0  
- YOLOv8 (Ultralytics, optional)  
- OpenCV  
- NumPy  
- Streamlit  

---

## 🏁 How to Run

### 1️⃣ Prerequisites

- Python 3.10 or higher  
- Git (optional, if cloning repo)  
- Recommended: Virtual environment (venv or conda)  

---

### 2️⃣ Clone the Repository

```bash
git clone git@github.com:dhanushiatus/wildlife_animal_classifier_ai.git
cd wildlife_animal_classifier_ai
```
Create a Virtual Environment (Optional)
```python -m venv venv```

Install Dependencies
```pip install -r requirements.txt```

Run the Streamlit App
```streamlit run app.py```

Using the App

Upload an Image – Click “Choose an image” and select a wildlife photo

YOLOv8 Detection (if enabled) – Animal is isolated automatically

Click “Classify Animal” – AI predicts the species

View Top-5 Predictions – Confidence levels:

🟢 >80% → High confidence

🟡 50–80% → Medium confidence

🔴 <50% → Low confidence
