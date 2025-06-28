# Traffic_signs_classification

https://drive.google.com/file/d/1qDtr3K7kDc0ZxlboeWVXLp4iMi5gujCP/view?usp=sharing

# 🛑 German Traffic Sign Classification with Deep Learning

This project demonstrates the development of a **Convolutional Neural Network (CNN)** to classify traffic signs 

The focus is on building a **robust, reliable model** that can accurately recognize 43 traffic sign classes, supporting safe decision-making in real-world applications such as autonomous vehicles.

---

## 🎯 Objectives

- Build a high-performance image classification model for traffic signs.
- Implement thorough preprocessing: grayscale conversion, histogram equalization, normalization.
- Address class imbalance and improve generalization through data augmentation.
- Evaluate model performance on training, validation, and test sets.
- Fine-tune hyperparameters to improve accuracy and reliability.

---

## ⚙️ Features

✅ **Custom Preprocessing Pipeline**
- Grayscale conversion to simplify input.
- Histogram equalization to normalize lighting.
- Image resizing and scaling.

✅ **Model Architecture**
- CNN inspired by LeNet:
  - Multiple convolutional layers.
  - Pooling layers for dimensionality reduction.
  - Dropout for regularization.
  - Dense layers with softmax output.

✅ **Training Strategy**
- Stratified split to preserve label distribution.
- Careful monitoring of loss and accuracy.
- Tuning batch size, learning rate, and epochs.

✅ **Data Augmentation**
- Rotation, zoom, width/height shift, shear transforms.

✅ **Evaluation**
- Clear metrics on validation and test sets to measure generalization.

---

## 📊 Results

After multiple iterations and fine-tuning, the final model achieved:

- **Training Accuracy:** ~96%
- **Validation Accuracy:** ~99%
- **Test Accuracy:** ~96%

This demonstrates strong generalization and suitability for real-world datasets.

---

## 🧪 Example Testing Observations

Initially, the model correctly predicted 3 out of 5 test images, highlighting areas for improvement.  
After hyperparameter tuning and data augmentation, the model classified **all test samples correctly**, showcasing significantly improved robustness.

---

## 🧠 Future Improvements

- Experiment with transfer learning (e.g., pretrained ResNet or EfficientNet).
- Apply k-fold cross-validation.
- Explore ensemble methods for further accuracy gains.
- Deploy the model in an interactive web application (e.g., Streamlit).

---

## 💻 Requirements

Below are the main Python libraries used in this project:


