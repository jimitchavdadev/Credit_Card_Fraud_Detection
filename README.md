# Credit Card Fraud Detection

## 📌 Objective

This project aims to detect fraudulent transactions in credit card data using machine learning techniques. The dataset is highly imbalanced, requiring special handling techniques such as undersampling and Synthetic Minority Over-sampling Technique (SMOTE). The model is trained using Logistic Regression, and various evaluation metrics are used to assess its performance.

---

## 📂 Dataset

**Dataset Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
The dataset contains transactions made by European cardholders in September 2013. It has 284,807 transactions, with only 492 cases of fraud (0.172%).

### Features

- **Time:** Seconds elapsed between the transaction and the first transaction in the dataset.
- **V1 - V28:** Principal components obtained via PCA (to maintain confidentiality).
- **Amount:** Transaction amount.
- **Class:** 0 (Legitimate) / 1 (Fraudulent).

---

## 🛠 Implementation

### 1️⃣ **Setup & Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
```

### 2️⃣ **Load and Explore Dataset**

```python
data = pd.read_csv('creditcard.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())
```

### 3️⃣ **Class Distribution**

```python
print(data['Class'].value_counts())
```

Fraudulent transactions are significantly fewer, requiring balancing techniques.

### 4️⃣ **Balancing Data** (Undersampling)

```python
fraud = data[data.Class == 1]
legit = data[data.Class == 0].sample(n=len(fraud), random_state=42)
balanced_data = pd.concat([fraud, legit])
```

### 5️⃣ **Feature Scaling**

```python
scaler = StandardScaler()
X = balanced_data.drop(columns=['Class'])
Y = balanced_data['Class']
X_scaled = scaler.fit_transform(X)
```

### 6️⃣ **Train-Test Split**

```python
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)
```

### 7️⃣ **Model Training**

```python
model = LogisticRegression(max_iter=10000, solver='liblinear', C=1.0, random_state=42)
model.fit(X_train, Y_train)
```

### 8️⃣ **Model Evaluation**

```python
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
```

### 9️⃣ **Confusion Matrix**

```python
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### 🔟 **ROC Curve**

```python
Y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### 1️⃣1️⃣ **Save Model**

```python
import joblib
joblib.dump(model, 'credit_card_fraud_model.pkl')
```

---

## 📊 Results & Insights

- The dataset was highly imbalanced; undersampling was used to balance it.
- Logistic Regression provided decent performance, but other models like Random Forest or XGBoost could improve results.
- Visualization techniques (PCA, ROC Curve, Confusion Matrix) help interpret model performance.

---

## 🚀 Future Enhancements

- Use **SMOTE** instead of undersampling for better data representation.
- Try **other ML models** (e.g., Random Forest, XGBoost, Neural Networks).
- Implement **real-time fraud detection** using streaming frameworks.

---

## 🤝 Contribution

Feel free to open an issue or submit a pull request if you have suggestions!

---

## 📜 License

This project is licensed under the MIT License.

---

### 🎯 Credits

Dataset provided by **Machine Learning Group - ULB** on Kaggle.
