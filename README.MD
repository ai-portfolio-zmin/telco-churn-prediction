Telco Customer Churn Prediction

Predicting customer churn for a telecom company using machine learning.  
This project builds a baseline predictive model (Logistic Regression) to identify customers likely to discontinue their service.

---

Overview
Customer churn is one of the most important metrics for subscription-based businesses.  


The project walks through a complete ML workflow:
- Data cleaning & preprocessing  
- Feature encoding  
- Model training (Logistic Regression)  
- Evaluation and interpretation  

This serves as a **baseline project** before exploring ensemble models and advanced evaluation techniques.

---

Dataset
**Telco Customer Churn** dataset from  - https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- 7,043 customers  
- Features: demographics, service subscriptions, billing, contract type  
- Target: `Churn` (Yes / No)

> The dataset is **not included** in this repository due to licensing.  
> Download it from Kaggle  before running the notebook.

---

Methods
1. Data cleaning and type conversion  
2. Handling missing values  
3. Encoding categorical variables with `pd.get_dummies()`  
4. Model training using Logistic Regression  
5. Model evaluation (accuracy)  
6. Basic interpretation of churn drivers  

---

Results
- Baseline model: Logistic Regression
- Evaluation metric: Accuracy = 80%  
- Key churn drivers (based on coefficients):
  - Month-to-month contracts → higher churn risk  
  - Short tenure → higher churn risk  
  - High charge → higher churn risk  

---

Next Steps
Planned improvements as learning progresses:
- Add advanced metrics: **ROC-AUC**, **Precision**, **Recall**, **F1-score**, **Confusion Matrix**  
- Compare multiple models: **Logistic Regression**, **Random Forest**, **XGBoost**  
- Integrate **Pipelines** for preprocessing  
- Add **feature importance / odds ratio** interpretation  
- Deploy a simple **Streamlit app** for prediction demo  