# Student Performance Prediction Using Machine Learning

## 📌 Project Overview

**Student Performance Prediction Using Machine Learning** is a Data Science project that analyzes student academic and behavioral information and predicts their final performance using Machine Learning regression techniques.

The project performs data preprocessing, Exploratory Data Analysis (EDA), feature preparation, model training, model comparison, evaluation, and prediction for new student data.

This project was developed as part of the **Data Science Major Project** for internship evaluation.

---

## 🎯 Problem Statement

Educational institutions collect large amounts of student academic data. Analyzing this data can help identify factors that influence student performance and predict future academic outcomes.

The objective of this project is to develop a Machine Learning system that can:

* Analyze student performance data
* Identify patterns and relationships in the dataset
* Preprocess and prepare data for Machine Learning
* Train regression models
* Evaluate model performance
* Compare multiple regression algorithms
* Predict the final score of a new student

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Load and analyze a student performance dataset.
2. Perform data cleaning and preprocessing.
3. Perform Exploratory Data Analysis (EDA).
4. Identify relationships between student characteristics and performance.
5. Train regression Machine Learning models.
6. Evaluate the trained models using standard metrics.
7. Select the best-performing model.
8. Predict the performance of new students.
9. Generate visualizations and outputs for analysis.
10. Prepare a complete Data Science project workflow.

---

## 🛠️ Technologies Used

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Programming language                      |
| Pandas           | Data manipulation and analysis            |
| NumPy            | Numerical computation                     |
| Scikit-learn     | Machine Learning                          |
| Matplotlib       | Data visualization                        |
| Seaborn          | Statistical visualization                 |
| Jupyter Notebook | Data analysis and experimentation         |
| Pickle           | Saving the trained Machine Learning model |

---

## 📊 Dataset

The project uses a student performance dataset containing **1,000 student records**.

### Dataset Features

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| `student_id`       | Unique student identifier                   |
| `gender`           | Student gender                              |
| `age`              | Student age                                 |
| `study_hours`      | Average study hours per day                 |
| `attendance`       | Attendance percentage                       |
| `previous_score`   | Previous academic score                     |
| `assignment_score` | Assignment score                            |
| `midterm_score`    | Midterm examination score                   |
| `extracurricular`  | Participation in extracurricular activities |
| `internet_access`  | Availability of internet access             |
| `sleep_hours`      | Average sleep hours per day                 |
| `final_score`      | Final academic score and prediction target  |

The target variable used for prediction is:

```text
final_score
```

---

## 🔄 Project Workflow

```text
Student Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Preparation
      │
      ▼
Train/Test Split
      │
      ▼
Machine Learning Models
      │
      ├── Linear Regression
      │
      └── Random Forest Regression
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
New Student Data
      │
      ▼
Performance Prediction
```

---

## 🔍 Exploratory Data Analysis

The project performs several EDA operations to understand student performance.

### Visualizations

* Final score distribution
* Study hours vs final score
* Attendance vs final score
* Previous score vs final score
* Correlation heatmap
* Actual vs predicted scores
* Model comparison

These visualizations help identify relationships between academic factors and final performance.

---

## 🤖 Machine Learning Models

Two regression algorithms are trained and compared.

### 1. Linear Regression

Linear Regression is used as the baseline regression model to predict the student's final score based on the available features.

### 2. Random Forest Regression

Random Forest Regression is used as a second model to compare its predictive performance with Linear Regression.

---

## 📈 Model Evaluation

The models are evaluated using:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

Lower MAE indicates better performance.

### Mean Squared Error — MSE

Measures the average squared difference between actual and predicted values.

Lower MSE indicates better performance.

### Root Mean Squared Error — RMSE

RMSE measures the standard deviation of prediction errors.

Lower RMSE indicates better performance.

### R² Score

R² measures how well the model explains the variation in the target variable.

Higher R² indicates better performance.

---

## 🏆 Model Results

Based on the current dataset and training configuration:

| Model                 |        MAE |        MSE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: | ---------: |
| **Linear Regression** | **2.4237** | **9.5411** | **3.0889** | **0.8681** |
| Random Forest         |     2.9433 |    13.8944 |     3.7275 |     0.8080 |

### Best Model

**Linear Regression**

```text
R² Score: 0.8681
MAE: 2.4237
RMSE: 3.0889
```

The Linear Regression model achieved the highest R² score among the tested models and was therefore selected as the final prediction model.

> Note: Exact metric values may vary slightly depending on the dataset version and installed library versions.

---

## 🔮 Student Performance Prediction

The trained model can predict the final performance of a new student.

Example input:

```text
Gender: Female
Age: 18
Study Hours: 5
Attendance: 90%
Previous Score: 78
Assignment Score: 85
Midterm Score: 82
Extracurricular: Yes
Internet Access: Yes
Sleep Hours: 7
```

The system processes these inputs and generates a predicted final score.

Example:

```text
Predicted Final Score: XX.XX / 100
Performance Category: Very Good
```

The actual prediction is generated dynamically by the trained Machine Learning model.

---

## 📁 Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   └── prediction.py
│
├── models/
│   └── student_performance_model.pkl
│
├── outputs/
│   ├── graphs/
│   │   ├── final_score_distribution.png
│   │   ├── study_hours_vs_score.png
│   │   ├── attendance_vs_score.png
│   │   ├── previous_score_vs_final_score.png
│   │   ├── correlation_heatmap.png
│   │   ├── actual_vs_predicted.png
│   │   └── model_comparison.png
│   │
│   └── predictions/
│       └── model_results.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone or download the project and navigate to the project directory.

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```powershell
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the complete project using:

```bash
python main.py
```

The system provides the following options:

```text
1. Run Exploratory Data Analysis (EDA)
2. Train Machine Learning Models
3. Predict Student Performance
4. Run Complete Project
5. Exit
```

### Run EDA

Select:

```text
1
```

### Train Models

Select:

```text
2
```

### Predict Student Performance

Select:

```text
3
```

### Run Complete Project

Select:

```text
4
```

---

## 📓 Jupyter Notebook

The complete analysis is also available in:

```text
notebooks/student_performance_analysis.ipynb
```

The notebook contains:

* Dataset loading
* Dataset inspection
* Data preprocessing
* EDA
* Visualization
* Model training
* Model evaluation
* New student prediction

To launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📸 Required Project Screenshots

The project generates outputs that can be used for the Major Project documentation.

Recommended screenshots include:

1. Dataset preview
2. Dataset information
3. Missing-value analysis
4. Final score distribution
5. Study hours vs final score
6. Attendance vs final score
7. Correlation heatmap
8. Model training output
9. Model comparison
10. Actual vs predicted graph
11. New student prediction output

---

## 📌 Results

The project demonstrates that Machine Learning can be used to predict student academic performance based on academic, behavioral, and lifestyle-related features.

Among the tested models, **Linear Regression achieved the highest R² score of approximately 0.868**, making it the selected model for final prediction.

---

## 🚀 Future Enhancements

Possible future improvements include:

* Adding more student-related features
* Testing additional Machine Learning algorithms
* Hyperparameter tuning
* Cross-validation
* Building an interactive Streamlit dashboard
* Adding student performance classification
* Creating automated performance reports
* Deploying the prediction system as a web application
* Adding database integration
* Monitoring model performance over time

---

## 🎓 Academic Project

This project was developed as part of the **Data Science Major Project** for internship evaluation.

### Project Type

```text
Data Science / Machine Learning
```

### Prediction Type

```text
Regression
```

### Target Variable

```text
final_score
```

### Primary Algorithm

```text
Linear Regression
```

---

## 👨‍💻 Author

**Singipurapu Dinesh**

Computer Science Student | Python Developer | AI & Data Science Enthusiast

---

## 📄 License

This project is intended for educational and academic purposes.
