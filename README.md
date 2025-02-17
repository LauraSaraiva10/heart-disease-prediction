# Heart Disease Prediction

## 📌 Project Overview
This project aims to predict heart disease using machine learning models. It evaluates multiple classification algorithms and applies hyperparameter tuning to improve performance. The dataset used is a structured heart disease dataset with relevant health indicators.

## 📂 Project Structure
```
heart-disease-prediction/
├── data/               # Store the dataset (heart.csv)
├── notebooks/          # Jupyter notebooks for experimentation
├── src/                # Python scripts
│   ├── __init__.py     # Makes 'src' a package
│   ├── data_loader.py  # Functions to load and preprocess data
│   ├── train.py        # Model training scripts
│   ├── evaluate.py     # Model evaluation metrics
│   ├── hyperparameter.py # Hyperparameter tuning
│   ├── visualization.py # Data visualization functions
├── requirements.txt    # Dependencies
├── README.md           # Project description
└── main.py            # Main script to run the project
```

## 📊 Dataset
- **Source:** [heart.csv](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Target Variable:** `target` (1 = Disease, 0 = No Disease)
- **Features:** Age, Cholesterol, Blood Pressure, Max Heart Rate, etc.

## 🚀 Installation & Setup

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Main
Run the **main script** to train and evaluate models:
```sh
python main.py
```

## 🛠️ Features
✔ Multiple ML models: Random Forest, Naive Bayes, Gradient Boosting, KNN, Logistic Regression, SVM

✔ Data scaling for sensitive models

✔ Hyperparameter tuning using GridSearchCV

✔ Feature importance visualization

✔ ROC Curve and performance metrics

## 📈 Model Performance
- The best model is **Random Forest** after hyperparameter tuning.
- Model evaluation includes **accuracy, recall, and ROC AUC score**.
- Feature importance and correlation heatmaps help in understanding data.

## 📊 Visualizations
The project includes the following plots:
- **Feature Importance**
- **Correlation Heatmap**
- **ROC Curve**

## 🤝 Acknowledgments
- Inspiration: YouTube tutorial
- Libraries Used: `scikit-learn`, `matplotlib`, `seaborn`, `pandas`
