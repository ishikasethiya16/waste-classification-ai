# AI Waste Classification System ♻️

A complete end-to-end Deep Learning system that classifies waste into **Biodegradable** and **Recyclable** categories using CNNs and Transfer Learning.

## 🚀 Features
- **Dual Model Support**: Custom CNN and MobileNetV2 (Transfer Learning).
- **Interactive UI**: Streamlit dashboard for image upload and live webcam input.
- **Smart Suggestions**: Provides disposal advice for each category.
- **Training Analytics**: Visualizes Accuracy/Loss and performance metrics.
- **Real-time Dashboard**: Tracks classification statistics.

## 📁 Project Structure
- `dataset/`: Organizes images for training and validation.
- `models/`: Stores trained model files (`.h5`) and evaluation plots.
- `src/`: Core logic for model building, preprocessing, and utilities.
- `app.py`: The Streamlit web application.
- `train.py`: Script to train and compare models.
- `dataset_setup.py`: Utility to prepare folder structure and data.

## 🛠️ Getting Started

### 1. Installation
Ensure you have Python installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Run the setup script to create the folders:
```bash
python dataset_setup.py
```
**Important**: Download the [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) from Kaggle and place the images in `dataset/train/` and `dataset/val/` under the respective class folders.

### 3. Training
Train the models using:
```bash
python train.py
```
This will save `custom_cnn.h5` and `transfer_learning.h5` in the `models/` folder.

### 4. Running the Dashboard
Launch the Streamlit app:
```bash
streamlit run app.py
```

## 🧠 Model Details
- **Custom CNN**: 3 Convolutional layers with MaxPooling and Dropout.
- **Transfer Learning**: MobileNetV2 base with custom Dense layers (Frozen weights for faster training).

## 💡 Smart Suggestions
- **Biodegradable** → Compost (Green)
- **Recyclable** → Recycle bin (Blue)

---
Developed as a production-ready AI Waste Classification project.
