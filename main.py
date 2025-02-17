import sys
import os

sys.path.append(os.path.abspath("src"))

from src.data_loader import load_data, scale_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.hyperparameter import tune_random_forest
from src.visualization import plot_feature_importance, plot_correlation_heatmap
import pandas as pd

def main():

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("\n Scaling data (for scale-sensitive models)...")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    print("\n🛠️ Training models...")
    models = train_models(X_train, X_test, y_train)

    print("\n📈 Evaluating models...")
    evaluate_models(models, X_test, X_test_scaled, y_test)

    print("\n🔍 Performing hyperparameter tuning for Random Forest...")
    best_forest = tune_random_forest(X_train, y_train)

    print("\n✅ Best tuned Random Forest model trained:")
    print(best_forest)

    print("\n🌟 Evaluating the best model...")
    best_score = best_forest.score(X_test, y_test)
    print(f"Best Random Forest Model Accuracy: {best_score:.4f}")

    print("\n📊 Plotting Feature Importance...")
    plot_feature_importance(best_forest)

    print("\n🔥 Generating Correlation Heatmap...")
    df = pd.read_csv("data/heart.csv")
    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()