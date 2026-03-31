# %%
#Importing libraries
#pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# %%
df = pd.read_csv("loan_dataset.csv")

# %%
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
categorical_values = df.select_dtypes(include=["str"]).columns
numerical_values = df.select_dtypes(include=["number"]).columns

# %%
categorical_values

# %%
numerical_values

# %%
#Pre-processing
from sklearn.impute import SimpleImputer

num_inp = SimpleImputer(strategy='mean')
df[numerical_values] = num_inp.fit_transform(df[numerical_values])


str_inp = SimpleImputer(strategy='most_frequent')
df[categorical_values] = str_inp.fit_transform(df[categorical_values])

# %%
df.head()

# %%
df.isnull().sum()

# %%
#EDA

classes_count = df["Loan_Approved"].value_counts()

plt.pie(classes_count, labels=["No","Yes"], autopct="%1.1f%%")
plt.title("Is loan approved or not")
plt.show()

# %%
#Gender ratio
gender_count = df["Gender"].value_counts()
ax = sns.barplot(gender_count)
ax.bar_label(ax.containers[0])

# %%
#Education ratio
edu_count = df["Education_Level"].value_counts()
ax = sns.barplot(edu_count)
ax.bar_label(ax.containers[0])



# %%
#Analyze income

sns.histplot(
    data = df,
    x="Applicant_Income",
    bins=20
)

# %%
sns.histplot(
    data = df,
    x="Coapplicant_Income",
    bins=20
)

# %%
fig, axes = plt.subplots(2,3)

sns.boxplot(ax=axes[0,0], data=df, x="Loan_Approved", y="Applicant_Income")
sns.boxplot(ax=axes[0,1], data=df, x="Loan_Approved", y="Credit_Score")
sns.boxplot(ax=axes[0,2], data=df, x="Loan_Approved", y="DTI_Ratio")
sns.boxplot(ax=axes[1,0], data=df, x="Loan_Approved", y="Savings")
sns.boxplot(ax=axes[1,1], data=df, x="Loan_Approved", y="Loan_Amount") 
sns.boxplot(ax=axes[1,2], data=df, x="Loan_Approved", y="Age") 

plt.tight_layout()

# %%
sns.histplot(
    data=df,
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)

# %%
df = df.drop(columns=["Applicant_ID"])

# %%
df.head()

# %%
df.columns
df.info()

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
df["Education_Level"] = le.fit_transform(df["Education_Level"])
      


# %%
df.head()

# %%
cols = ["Employment_Status", "Marital_Status","Loan_Purpose", "Property_Area","Gender","Employer_Category"]

ohe = OneHotEncoder(drop="first",sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# %%
df.info()

# %%
# Corealtion matrix

num_col = df.select_dtypes(include='number')
corr_mat = num_col.corr()

plt.figure(figsize=(15, 8))

sns.heatmap(
    corr_mat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

# %%
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# %%
X

# %%
y

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42)

# %%
X_train

# %%
y_train

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_train_scaled

# %%
X_test_scaled

# %%
#Model train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)
y_pred_train = log_model.predict(X_train_scaled)

print("Logistic model")
print("Precision Score: ",precision_score(y_test, y_pred))
print("Recall Score: ",recall_score(y_test, y_pred))
print("F1 Score: ",f1_score(y_test, y_pred))
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Accuracy Score train: ",accuracy_score(y_train, y_pred_train))
print("Confusion_matrix : ",confusion_matrix(y_test, y_pred))

# %%
#Improving accuracy

#Random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    class_weight= 'balanced',
    n_estimators=401,
    oob_score=True,
    max_depth=4
    
)

rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
y_pred_train = log_model.predict(X_train_scaled)

print("OOB score :", rf.oob_score_ *100)
print("Accurcy score:",accuracy_score(y_test, y_pred)*100)

print("Random forest model")
print("Precision Score: ",precision_score(y_test, y_pred))
print("Recall Score: ",recall_score(y_test, y_pred))
print("F1 Score: ",f1_score(y_test, y_pred))
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Accuracy Score train: ",accuracy_score(y_train, y_pred_train))
print("Confusion_matrix : ",confusion_matrix(y_test, y_pred))

# %%
from sklearn.svm import SVC
model = SVC(kernel='rbf')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
from sklearn.ensemble import  VotingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

rf = RandomForestClassifier(
    class_weight= 'balanced',
    n_estimators=401,
    oob_score=True,
    max_depth=4   
)

SVC = SVC(probability=True)

lr = LogisticRegression(max_iter=10000)

voting_model = VotingClassifier(estimators=[
    ('lr', lr),
    ('svc', SVC),   
    ('rfc', rf),
], voting='soft')


# %%
voting_model.fit(X_train_scaled, y_train)
y_pred = voting_model.predict(X_test_scaled)

accuracy_score(y_test, y_pred)

# %%
import xgboost as xgb
from sklearn.metrics import classification_report
xgb_clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

xgb_clf.fit(X_train_scaled, y_train)

y_pred = xgb_clf.predict(X_test_scaled)

print("Acc",accuracy_score(y_test, y_pred)*100, "%")
print("Clf report",classification_report(y_test, y_pred))

# %%
from lightgbm import LGBMClassifier

cat = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.1,
    scale_pos_weight=3,
    random_state=42
)

cat.fit(X_train_scaled, y_train)
y_pred = cat.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))

# %%
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    auto_class_weights='Balanced',  # handles imbalance automatically
    verbose=0,                       # silences training logs
    random_state=42
)

cat.fit(X_train_scaled, y_train)
y_pred = cat.predict(X_test_scaled)
print("Accuracy score",accuracy_score(y_test, y_pred)*100,"%")

# %%
#Stacking classifier
from sklearn.ensemble import StackingClassifier

lr = LogisticRegression(max_iter=5000)

rf = RandomForestClassifier(
    class_weight= 'balanced',
    n_estimators=500,
    oob_score=True,
    max_depth=4   
)

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

meta_model = LogisticRegression(max_iter=5000)

stacking_model = StackingClassifier(estimators=[
        ('lr', lr),
        ('rf',rf),
        ("xgb",xgb_clf)
        ],
        final_estimator = meta_model,
        cv=5
)

# %%
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

# %%
from sklearn.metrics import classification_report

print("Accuracy score: ",accuracy_score(y_test, y_pred)*100)
print("Classification Report:\n ",classification_report(y_test, y_pred))


