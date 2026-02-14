# 🧠 Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
This project implements a Deep Learning model using an Artificial Neural Network (ANN) to predict customer churn. The model analyzes customer demographic and banking data to determine whether a customer is likely to leave the bank.

An interactive Streamlit web application is included to allow real-time churn prediction based on user input.

---

## 🚀 Key Features
- Data preprocessing (Label Encoding, One-Hot Encoding, Feature Scaling)
- ANN model built using TensorFlow/Keras
- Binary classification using Sigmoid activation
- Model saved in `.h5` format
- Encoders and scaler stored using `pickle`
- Interactive Streamlit web app
- Real-time churn probability output

---

## 📂 Project Structure

ANN-Classification-Churn/
- Churn_Modelling.csv        → Dataset  
- experiments.ipynb          → Model training notebook  
- prediction.ipynb           → Prediction testing notebook  
- model.h5                   → Trained ANN model  
- scaler.pkl                 → StandardScaler object  
- label_encoder_gender.pkl   → Label encoder for gender  
- onehot_encoder_geo.pkl     → OneHot encoder for geography  
- app.py                     → Streamlit web application  
- requirements.txt           → Required libraries  

---

## 🧾 Dataset Details
The dataset contains 10,000 customer records with the following features:

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

Target Variable:
- Exited (0 = Customer stays, 1 = Customer leaves)

---

## 🏗️ Model Architecture
- Input Layer
- 2 Hidden Layers (ReLU activation)
- Output Layer (Sigmoid activation)
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy

---

## 🔄 Project Workflow
1. Data Cleaning & Feature Selection  
2. Encoding Categorical Variables  
3. Feature Scaling using StandardScaler  
4. Train-Test Split  
5. ANN Model Building using TensorFlow/Keras  
6. Model Training & Evaluation  
7. Saving Model and Preprocessing Objects  
8. Deployment using Streamlit  

---

## 💻 Installation & Setup

### 1. Clone Repository
git clone https://github.com/krishnaik06/ANN-CLassification-Churn.git  
cd ANN-CLassification-Churn  

### 2. Install Dependencies
pip install -r requirements.txt  

### 3. Run the Application
streamlit run app.py  

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Streamlit  
- Pickle  

---

## 📈 Model Outcome
- Predicts probability of customer churn  
- Helps identify high-risk customers  
- Useful for retention strategy planning  

---

## 🎯 Business Use Case
Financial institutions can use this model to:
- Identify customers likely to churn  
- Improve customer retention  
- Optimize marketing efforts  
- Increase long-term profitability  

---
