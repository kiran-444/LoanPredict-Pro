# 🏦 Loan Approval Prediction System

## 📌 Overview

This is a **full-stack Loan Approval Prediction System** that combines a high-accuracy Machine Learning model with an interactive web interface built using HTML, CSS, and JavaScript. Users can enter their applicant details directly into the UI and receive an instant loan eligibility prediction powered by the backend ML model.

The core ML focus was on achieving **high precision for the loan rejection class (Class 0)** — ensuring genuine rejections are correctly identified, reducing the risk of approving high-risk applicants.

The final **Stacking Ensemble Classifier** achieved:

| Metric | Score |
|---|---|
| **Overall Accuracy** | **93.5%** |
| Precision (Rejected – Class 0) | **0.98** |
| Recall (Rejected – Class 0) | 0.93 |
| F1-Score (Rejected – Class 0) | 0.95 |
| Precision (Approved – Class 1) | 0.85 |
| Recall (Approved – Class 1) | 0.95 |
| F1-Score (Approved – Class 1) | 0.90 |

> ✅ The model achieves **98% precision on loan rejections**, meaning when it flags an application as rejected, it is almost certainly correct — a critical requirement in real-world lending scenarios.

---

## 🚀 Tech Stack

### 🧠 Machine Learning
* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

### 🌐 Web Application
* Flask (Python backend)
* HTML5 & CSS3 (Frontend UI)
* JavaScript (Form handling & API calls)

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

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest Classifier | Strong ensemble base |
| Support Vector Machine (SVM) | Kernel-based classifier |
| Voting Classifier | Soft voting ensemble |
| XGBoost Classifier | Gradient boosting |
| LightGBM Classifier | Efficient boosting |
| CatBoost Classifier | Categorical-aware boosting |
| **Stacking Classifier** | **Best performer — final model** |

---

## 📊 Model Evaluation (Final Model)

```
Accuracy: 93.5%

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.93      0.95       139   ← Loan Rejected
           1       0.85      0.95      0.90        61   ← Loan Approved

    accuracy                           0.94       200
   macro avg       0.92      0.94      0.93       200
weighted avg       0.94      0.94      0.94       200
```

**Key Insight:** The model's **precision of 0.98 for Class 0 (Rejected)** means nearly all flagged rejections are true rejections — minimizing the risk of incorrectly approving high-risk loan applications.

---

## 🖥️ Web Interface

The project includes a fully functional **web-based prediction interface** built with HTML, CSS, and JavaScript, connected to a Flask backend that loads the trained model and returns predictions in real time.

### UI Preview

![Loan Prediction UI](./static/images/ui.png)

### Input Fields Covered

**💰 Financial Details**
* Applicant Income
* Co-Applicant Income
* Credit Score (300–850)
* Loan Amount
* Loan Term (Months)
* DTI Ratio

**👤 Applicant Profile**
* Age
* Dependents
* Existing Loans
* Collateral Value
* Savings
* Education Level

**🏷️ Classifications**
* Employment Type
* Marital Status
* Loan Purpose
* Property Area
* Gender
* Employer Type

> The **"Analyze Application"** button sends the form data to the Flask API, which preprocesses the input (scaling + encoding) and returns a prediction from the stacking ensemble model.

---

## 📦 Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost flask
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

### Step 3: Train the model (generates .pkl files)

```bash
python loan_approval.py
```

### Step 4: Start the Flask backend

```bash
cd backend
python app.py
```

### Step 5: Open the frontend

Open `frontend/index.html` in your browser, fill in the applicant details, and click **Analyze Application**.

---

## 📁 Project Structure

```
├── 📁 app
│   └── 🐍 app.py
├── 📁 data
│   ├── 📁 processed
│   │   └── 📄 loan_data_cleaned.csv
│   └── 📁 raw
│       └── 📄 loan_dataset.csv
├── 📁 models
├── 📁 notebooks
│   ├── 📄 01_EDA and Data_preprocessing.ipynb
│   ├── 📄 02_Feature_engineering.ipynb
│   └── 📄 03_Model_training.ipynb
├── 📁 report
├── 📁 static
│   ├── 📁 css
│   │   └── 🎨 style.css
│   ├── 📁 images
│   │   └── 🖼️ ui.png
│   └── 📁 js
│       └── 📄 script.js
├── 📁 templates
│   └── 🌐 index.html
├── ⚙️ .gitignore
├── 📝 README.md
└── 📄 requirements.txt
```

---

## 💡 Key Highlights

✔ End-to-end Machine Learning pipeline  
✔ Focused on **precision for loan rejection** — business-critical metric  
✔ Multiple model comparison with systematic evaluation  
✔ Ensemble learning (Voting & Stacking) for maximum performance  
✔ **Interactive web UI** built with HTML, CSS & JavaScript  
✔ **Flask REST API** serving real-time predictions  
✔ Data visualization & insights  
✔ Real-world dataset handling  

---

## 🎯 Future Improvements

* Deploy on cloud (Render / AWS / Heroku) for public access
* Perform hyperparameter tuning with GridSearchCV / Optuna
* Add prediction confidence score display in the UI
* Explore Deep Learning models (Neural Networks)
* Add explainability with SHAP values

---

## 👩‍💻 Author

**Kiran Metri**  
Final Year CSE Student

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!