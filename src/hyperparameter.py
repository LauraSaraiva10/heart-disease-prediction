from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_data

def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=9), param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_