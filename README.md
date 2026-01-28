# Diabetes Prediction Project 🩺

This project aims to predict the presence of diabetes in patients based on several diagnostic measurements. It compares two ensemble learning approaches: **Random Forest** and **AdaBoost**, utilizing automated machine learning pipelines for data preprocessing and hyperparameter tuning.

## 📊 Dataset Description
The dataset contains information about several health indicators:
* **Pregnancies**: Number of times pregnant.
* **Glucose**: Plasma glucose concentration.
* **BloodPressure**: Diastolic blood pressure (mm Hg).
* **SkinThickness**: Triceps skin fold thickness (mm).
* **Insulin**: 2-Hour serum insulin (mu U/ml).
* **BMI**: Body mass index.
* **DiabetesPedigreeFunction**: Genetic risk score.
* **Age**: Age in years.
* **BMI_Insulin_Ratio**: A custom feature created by multiplying BMI and Insulin.

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy)
- **Scikit-Learn** (Pipelines, GridSearchCV, Imputing, Scaling)
- **Visualization**: Seaborn, Matplotlib

## 🏗️ Workflow

### 1. Feature Engineering & Preprocessing
- **Custom Feature**: Added `BMI_Insulin_Ratio` to capture the interaction between body mass and insulin levels.
- **Handling Missing Values**: Used `SimpleImputer` with a **median** strategy. Specifically handled "0" values in vital columns which represent missing data.
- **Scaling**: Applied `StandardScaler` to normalize features for better model performance.

### 2. Machine Learning Pipelines
I implemented two distinct pipelines with `GridSearchCV` for hyperparameter optimization:

- **Pipeline 1: Random Forest Classifier**
  - Optimized: `n_estimators`, `max_depth`, and `min_samples_split`.
- **Pipeline 2: AdaBoost Classifier**
  - Used `DecisionTreeClassifier` as base estimators.
  - Optimized: `learning_rate` and `n_estimators`.

### 3. Model Evaluation
The models were evaluated using:
- Accuracy Score
- Precision, Recall, and F1-Score
- Correlation Matrix Heatmap
- Visual comparison of Correct vs. Incorrect predictions

## 🧪 Testing
The project includes a `run_simple_tests` function to ensure:
- The custom feature is correctly added.
- The training set maintains the correct input shape (9 features).
- The model outputs are valid binary classifications (0 or 1).

## 📈 Results
The project demonstrates how ensemble methods and proper preprocessing (handling zeros and scaling) significantly impact the predictive power of the model.
