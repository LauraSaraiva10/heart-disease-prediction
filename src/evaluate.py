from sklearn.metrics import recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, X_test_scaled, y_test):
    for name, model in models.items():
        X_used = X_test_scaled if name in ["KNN", "LogisticRegression", "SVM"] else X_test
        y_preds = model.predict(X_used)

        score = model.score(X_used, y_test)
        print(f"{name} Score: {score:.4f}")

        recall = recall_score(y_test, y_preds)
        print(f"{name} Recall Score: {recall:.4f}")

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_used)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            plt.plot(fpr, tpr, label=f"{name} (AUC: {roc_auc_score(y_test, y_probs):.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from data_loader import load_data, scale_data
    from train import train_models

    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    models = train_models(X_train, X_test, y_train)
    evaluate_models(models, X_test, X_test_scaled, y_test)