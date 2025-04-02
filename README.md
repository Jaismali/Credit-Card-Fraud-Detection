# Credit-Card-Fraud-Detection
💳 Credit Card Fraud Detection using Machine Learning  This project implements a machine learning model to detect fraudulent transactions from a dataset of 284K+ credit card transactions. It applies data preprocessing, feature selection, and class balancing techniques to improve fraud detection accuracy.

🔍 Overview

Fraud detection is a critical challenge in the financial sector. This project aims to detect fraudulent credit card transactions using machine learning techniques. By preprocessing data effectively and utilizing multiple classification models, we can improve the accuracy of fraud detection.

📌 Features

Data preprocessing using MinMaxScaler.

Training multiple machine learning models, including Logistic Regression, SVM, and XGBoost.

Performance evaluation using precision, recall, F1-score, and confusion matrix.

📂 Dataset

The dataset used is from Kaggle's Credit Card Fraud Detection Dataset.

It contains 284,807 transactions, with only 0.17% fraud cases.

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/jaismali/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Notebook

Open Fraud_Detection_Hybrid_Model.ipynb and run the cells sequentially.

🛠️ Tech Stack

Python (pandas, numpy, scikit-learn, seaborn, matplotlib, XGBoost)

Jupyter Notebook

🔬 Exploratory Data Analysis (EDA)

EDA is performed to visualize the data distribution and correlations:

# Correlation Heatmap
sns.heatmap(credit_card_data.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.show()

# Pair Plot
sample_data = credit_card_data.sample(1000)
sns.pairplot(sample_data, hue='Class', diag_kind='kde')
plt.show()

📊 Model Training

We train different models to improve fraud detection accuracy:

from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

📈 Model Evaluation

Instead of just checking accuracy, we focus on precision-recall metrics:

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, model.predict(X_test)))

🎯 Key Takeaways

MinMaxScaler ensures better feature scaling for imbalanced datasets.

SVM and XGBoost outperform Logistic Regression in detecting fraud cases.

Precision and Recall are prioritized over Accuracy for better fraud detection.

📜 License

This project is open-source under the MIT License
