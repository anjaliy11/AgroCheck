# AgroCheck 🌱  
_A Deep Learning-Based Potato Disease Classification System_

---

## 📌 Project Overview
AgroCheck is a **Deep Learning project** designed to classify **potato plant diseases** using image recognition.  
The model leverages Convolutional Neural Networks (CNNs) to detect and distinguish between **healthy potato leaves** and those affected by common diseases such as:

- **Early Blight**  
- **Late Blight**  
- **Healthy**

This project aims to assist **farmers and agricultural experts** in **early detection of crop diseases**, enabling timely intervention and improved crop yield.

---

## 🚀 Features
- 📷 **Image-based classification** of potato leaves  
- 🧠 **CNN-based deep learning model** trained on the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset)  
- 📊 **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score  
- 📈 Supports **real-time predictions** (future extension with Flask/Streamlit)  
- 📝 Well-structured dataset preprocessing, training, and evaluation pipeline  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- **TensorFlow / Keras** – Model building and training  
- **NumPy, Pandas** – Data handling  
- **Matplotlib, Seaborn** – Data visualization  
- **scikit-learn** – Metrics and model evaluation  

---
## 📂 Repository Structure
AgroCheck/
│── data/     
│── models/     
│── notebooks/       
│── src/        
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── evaluate.py
│── README.md        
│── requirements.txt     

yaml

---

## ⚙️ Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/anjaliy11/AgroCheck.git
   cd AgroCheck
## Create a virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


## Install dependencies


pip install -r requirements.txt
Run training script



 ## python src/model_training.py
 ## 📊 Results
The CNN model achieves high accuracy on the PlantVillage potato dataset.
Example metrics:

Training Accuracy: ~97%

Validation Accuracy: ~95%

Test Accuracy: ~94%

(Exact results may vary depending on hyperparameters and dataset split.)

## 🔮 Future Work
✅ Deploy model with Flask/Streamlit for real-time predictions

✅ Extend to other crops & diseases

✅ Build a mobile application for farmers

✅ Optimize for edge devices (TensorFlow Lite)




