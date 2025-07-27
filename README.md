# MNIST Digit Classifier with PyTorch 🧠🖊️

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify handwritten digits from the MNIST dataset.

## 🚀 Features
- CNN model trained on MNIST with over 98% test accuracy
- Evaluation metrics with confusion matrix and accuracy plots
- Modular code structure (data, model, train, evaluate)
- Inference script for real-time digit prediction
- Easy to extend into a Streamlit web app for live demo

## 🧱 Technologies
- Python
- PyTorch
- Matplotlib / Seaborn / Scikit-learn

## 📸 Sample Output
- `images/train_loss.png`: Visualize training loss
- `images/confusion_matrix.png`: Model performance matrix

---

## 🧪 Run Locally

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python infer.py