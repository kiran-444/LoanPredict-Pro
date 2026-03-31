# 🏦 Loan Approval Prediction System

## 📌 Overview

This project predicts whether a loan application will be approved or not using Machine Learning algorithms. It covers the complete ML pipeline including data preprocessing, feature engineering, exploratory data analysis (EDA), and model building.

---

## 🚀 Tech Stack

* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

---

## 📂 Dataset

* File: `loan_dataset.csv`
* Contains applicant details such as:

  * Applicant Income
  * Coapplicant Income
  * Credit Score
  * Loan Amount
  * Employment Status
  * Property Area
  * Age
  * Loan Approval Status

---

## ⚙️ Project Workflow

### 🔹 Data Preprocessing

* Handling missing values using `SimpleImputer`
* Encoding categorical data using:

  * Label Encoding
  * One-Hot Encoding

### 🔹 Exploratory Data Analysis (EDA)

* Loan approval distribution (Pie Chart)
* Gender & Education analysis (Bar Charts)
* Income distribution (Histograms)
* Feature relationships (Boxplots)
* Correlation Heatmap

### 🔹 Feature Engineering

* Removed unnecessary columns
* Scaled features using `StandardScaler`

### 🔹 Model Building

The following models were implemented and compared:

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)
* Voting Classifier (Ensemble)
* XGBoost Classifier
* LightGBM Classifier
* CatBoost Classifier
* Stacking Classifier

---

## 📊 Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

## 📦 Required Libraries

The project uses the following Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* lightgbm
* catboost

### Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

---

## ▶️ How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the project

```bash
python loan_approval.py
```

---

## 📁 Project Structure

```
Mini_Project/
│
├── loan_approval.py
├── loan_dataset.csv
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 💡 Key Highlights

✔ End-to-end Machine Learning pipeline
✔ Multiple model comparison
✔ Ensemble learning (Voting & Stacking)
✔ Data visualization & insights
✔ Real-world dataset handling

---

## 🎯 Future Improvements

* Deploy using Flask or Streamlit
* Add user input interface
* Perform hyperparameter tuning
* Explore Deep Learning models

---

## 👨‍💻 Author

Kiran Metri
Final Year CSE Student

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
