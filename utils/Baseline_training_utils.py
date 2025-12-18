import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import ParameterGrid, GroupKFold
from tqdm import tqdm
import joblib
import os


def run_model(model_name, X_train_scale, X_train_rf, y_train, X_test_scaled, X_test_rf, y_test, path_name):
    X_train_used, X_test_used, model, param_grid, use_grid_search = choose_model(
        model_name, X_train_scale, X_test_rf, X_test_scaled, X_train_rf
    )

    results = None
    model_path = f"Models/{model_name}_{path_name}_best.joblib"
    results_path = f"Models/{model_name}_{path_name}_grid_results.joblib"

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model = joblib.load(model_path)
        print("Loaded model.")
        if os.path.exists(results_path):
            results = joblib.load(results_path)
    else:
        groups = X_train_used["participant_id"]

        if use_grid_search:
            print("Grid:", param_grid)
            model, results = grid_search(model, X_train_used, y_train, groups, param_grid)

        model = fit_full_model(model, X_train_used, y_train)

        os.makedirs("Models", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved trained model to {model_path}")

        if results is not None:
            joblib.dump(results, results_path)
            print(f"Saved grid search results to {results_path}")
    
    print("Evaluating on test set...")
    predictions, mae = evaluate_model(model, X_test_used, y_test)
    print(f"Model: {model_name} - Mean Absolute Error:", mae)

    return predictions, mae, results


def choose_model(model_name, X_train_scaled, X_test_rf, X_test_scaled, X_train_rf):
    if model_name == "linear":
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
        model = LinearRegression()
        use_grid_search = False  
        param_grid = None

    elif model_name == "lasso":
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
        model = Lasso()
        param_grid = {"alpha": [0.45, 0.5, 0.55, 0.6]}
        use_grid_search = True

    elif model_name == "random_forest":
        X_train_used = X_train_rf
        X_test_used = X_test_rf
        model = RandomForestRegressor()
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5]
        }
        use_grid_search = True

    else:
        raise ValueError("Unknown model name")

    return X_train_used, X_test_used, model, param_grid, use_grid_search


def grid_search(model, X_train_used, y_train, groups, param_grid):
    cv = GroupKFold(n_splits=4)

    param_combinations = list(ParameterGrid(param_grid))
    print("Total combinations:", len(param_combinations))
    best_score = np.inf
    best_params = None
    results = []

    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        model.set_params(**params)
        fold_scores = []

        for train_idx, val_idx in cv.split(X_train_used, y_train, groups=groups):
            X_tr = X_train_used.iloc[train_idx].drop(columns=["participant_id"])
            X_val = X_train_used.iloc[val_idx].drop(columns=["participant_id"])
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            # Conversion explicite en 1D
            y_val_arr = np.asarray(y_val).reshape(-1)

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae = np.mean(np.abs(preds - y_val_arr))
            fold_scores.append(mae)

        avg_score = np.mean(fold_scores)

        results.append({
            "params": params.copy(),
            "fold_scores": fold_scores.copy(),
            "avg_score": avg_score,
        })

        if avg_score < best_score:
            best_score = avg_score
            best_params = params.copy()

    print("Best CV score:", best_score)
    print("Best Hyperparameters:", best_params)
    best_model = model.__class__(**best_params)

    return best_model, results


def fit_full_model(model, X_train_used, y_train):
    X_train_no_id = X_train_used.drop(columns=["participant_id"])
    model.fit(X_train_no_id, y_train)
    return model


def evaluate_model(model, X_test_used, y_test):
    y_test_arr = np.asarray(y_test).reshape(-1)
    
    X_test_no_id = X_test_used.drop(columns=["participant_id"])
    print("Predicting on test set...")
    predictions = model.predict(X_test_no_id)
    mae = np.mean(np.abs(predictions - y_test_arr))
    print(f"Mean Absolute Error on test set: {mae:.3f}")
    return predictions, mae

    
def plot_true_vs_pred(y_true, y_pred, model_name="", pct=100, random_state=42):
    """
    pct : pourcentage de points à afficher (entre 0 et 100)
    """

    # Conversion en numpy au cas où
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Vérification du pourcentage
    if pct <= 0 or pct > 100:
        raise ValueError("pct must be in (0, 100]")

    # Nombre de points à afficher
    n_points = len(y_true)
    n_sample = int(n_points * (pct / 100))

    # Sous-échantillonnage aléatoire
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_points, size=n_sample, replace=False)

    y_true_sample = y_true[idx]
    y_pred_sample = y_pred[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.6)

    # ligne de prédiction parfaite
    min_val = min(np.min(y_true_sample), np.min(y_pred_sample))
    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    mae = np.mean(np.abs(y_pred - y_true))  # MAE calculé sur TOUT le dataset

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"{model_name} : y_true VS y_pred ({pct}% des points, MAE={mae:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_pred_density(y_true, y_pred, model_name="", pct=100, random_state=42):
    """
    Affiche y_true vs y_pred en carte de densité (hexbin) simple.
    - Pas de colorbar
    - Sous-échantillonnage optionnel avec pct
    - Pas de nettoyage / filtrage des données
    """

    # Conversion et mise en forme
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Vérification taille uniquement
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true et y_pred doivent avoir la même taille, "
                         f"reçu {y_true.shape[0]} et {y_pred.shape[0]}.")

    if not (0 < pct <= 100):
        raise ValueError("pct doit être dans l'intervalle (0, 100].")

    n = y_true.size
    n_sample = max(1, int(n * pct / 100))

    # Sous-échantillonnage, si pct<100
    if n_sample < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=n_sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    # Bornes communes pour un plot carré
    data_min = min(y_true.min(), y_pred.min())
    data_max = max(y_true.max(), y_pred.max())
    if data_max == data_min:
        data_max = data_min + 1.0  # éviter plage nulle

    margin = 0.05 * (data_max - data_min)
    x_min, x_max = data_min - margin, data_max + margin

    plt.figure(figsize=(6, 6))

    # Carte de densité simple 
    plt.hexbin(y_true, y_pred, gridsize=40, cmap="viridis", mincnt=1)

    # Ligne y = x
    plt.plot([x_min, x_max], [x_min, x_max], "--", linewidth=1, color="black")

    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"{model_name} : y_true vs y_pred\n"
              f"({pct}% of points)")

    plt.tight_layout()
    plt.show()


def plot_grid_search_results(results, param_name, model_name="", ax=None):
    rows = []
    for r in results:
        row = {**r["params"]}
        row["avg_score"] = r["avg_score"]
        rows.append(row)
    df = pd.DataFrame(rows)

    df_group = df.groupby(param_name)["avg_score"].mean().reset_index()
    df_group = df_group.sort_values(param_name)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(df_group[param_name], df_group["avg_score"], marker="o")
    ax.set_xlabel(param_name)
    ax.set_ylabel("MAE average")
    ax.set_title(f"{model_name} - {param_name}")
    ax.grid(True)


def plot_rf_grid_search(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    plot_grid_search_results(results, "n_estimators", "random_forest", ax=axes[0])
    plot_grid_search_results(results, "max_depth", "random_forest", ax=axes[1])
    plot_grid_search_results(results, "min_samples_split", "random_forest", ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_predictions_vs_time(time, y_pred, y_true, model_name, step=1):
    # Sous-échantillonnage
    time = time.iloc[::step]
    y_true = y_true.iloc[::step]
    y_pred = np.asarray(y_pred)[::step]

    # Tri par ordre temporel (sécurité)
    order = np.argsort(time.values)
    time_sorted = time.values[order]
    y_true_sorted = y_true.values[order]
    y_pred_sorted = y_pred[order]

    # ===== PLOT =====
    plt.figure(figsize=(14, 6))
    plt.plot(time_sorted, y_true_sorted, label="True heat flux", alpha=0.8)
    plt.plot(time_sorted, y_pred_sorted, label="Predicted heat flux", alpha=0.8)

    plt.title(f"{model_name} – Heat Flux vs Time")
    plt.xlabel("Time")
    plt.ylabel("Heat Flux")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

