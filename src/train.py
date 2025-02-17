from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data_loader import load_data, scale_data

def train_models(X_train, X_test, y_train):
    models = {
        "RandomForest": (RandomForestClassifier(random_state=9), False),
        "NaiveBayes": (GaussianNB(), False),
        "GradientBoosting": (GradientBoostingClassifier(), False),
        "KNN": (KNeighborsClassifier(), True),
        "LogisticRegression": (LogisticRegression(), True),
        "SVM": (SVC(probability=True), True)
    }

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    trained_models = {}

    for name, (model, is_scale_sensitive) in models.items():
        X_train_used = X_train_scaled if is_scale_sensitive else X_train
        model.fit(X_train_used, y_train)
        trained_models[name] = model
        print(f"{name} trained (Scale-Sensitive: {is_scale_sensitive})")

    return trained_models