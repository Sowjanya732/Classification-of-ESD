# Dermatology Disease Classification using Machine Learning  

This project applies machine learning techniques to classify dermatology diseases based on medical features. It uses feature selection, class imbalance handling, and model comparison to identify the best-performing classification model.

## Dataset  
The dataset contains dermatological attributes and class labels representing different skin diseases. It is preprocessed by handling missing values, standardization, and feature selection.  

## Workflow  
1. **Data Preprocessing**: Handling missing values, feature selection using RFECV  
2. **Class Imbalance Handling**: Applied SMOTEENN to balance dataset  
3. **Model Training**: Compared Logistic Regression, Random Forest, XGBoost, etc.  
4. **Hyperparameter Tuning**: Used RandomizedSearchCV to optimize performance  
5. **Model Evaluation**: Measured accuracy, precision, recall, and F1-score  

## Installation  
To run this project, install the required dependencies:  
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap seaborn matplotlib

## Usage  
Clone the repository and run the `ESD.py` script in Python:

git clone https://github.com/your-username/your-repository.git
cd your-repository
python ESD.py


## Results  
The model performances are as follows:  

| Model                  | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|------------------------|---------------|--------------|-----------|--------|----------|
| **SVM**               | 0.9980        | **1.0000**   | 1.0000    | 1.0000 | 1.0000   |
| Logistic Regression   | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| Gradient Boosting     | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| Decision Tree         | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| XGBoost              | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| MLP                  | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| Stacking Classifier  | 1.0000        | 0.9919       | 0.9923    | 0.9919 | 0.9919   |
| Random Forest        | 0.9980        | 0.9839       | 0.9847    | 0.9839 | 0.9836   |
| KNN                  | 0.9960        | 0.9839       | 0.9854    | 0.9839 | 0.9838   |
| Naïve Bayes          | 0.9313        | 0.9355       | 0.9546    | 0.9355 | 0.9319   |

**Best Model:** The **SVM** classifier achieved **100% test accuracy**.


