import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model):
    feature_importances = model.feature_importances_
    features = model.feature_names_in_

    sorted_idx = np.argsort(feature_importances)
    sorted_features = features[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    colors = plt.cm.YlGn(sorted_importances / max(sorted_importances))
    plt.barh(sorted_features, sorted_importances, color=colors)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance Plot")
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="YlGn", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()