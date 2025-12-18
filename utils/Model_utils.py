############### Imports ###############

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    RandomizedSearchCV,
    ParameterGrid,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import joblib

from utils.Baseline_training_utils import *

############### Basic featuring manipulation helpers ###############

def normalize_selected_columns(X_train, X_test, cols_to_normalize):
    """
    Standardize selected columns using stats fitted on X_train only (prevents leakage).

    Returns:
      - X_train_norm, X_test_norm: scaled copies of inputs
      - scaler: fitted StandardScaler (reuse for inference)
    """
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    scaler = StandardScaler()
    scaler.fit(X_train_norm[cols_to_normalize])

    X_train_norm[cols_to_normalize] = scaler.transform(X_train_norm[cols_to_normalize])
    X_test_norm[cols_to_normalize] = scaler.transform(X_test_norm[cols_to_normalize])

    return X_train_norm, X_test_norm, scaler


def drop_non_finite_rows(df):
    """
    Remove rows containing NaN / +inf / -inf across ALL columns.
    """
    mask_finite = np.all(np.isfinite(df.to_numpy()), axis=1)
    df_clean = df[mask_finite].copy()

    removed_count = len(df) - len(df_clean)
    print(f"Removed {removed_count} rows containing NaN / inf / -inf.")

    return df_clean, removed_count


def drop_non_finite_rows_numeric(X):
    """
    Remove rows where any *numeric* column contains NaN / +inf / -inf.
    Non-numeric columns (ex: participant_id) are ignored.
    """
    X = X.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        print("No numeric columns found, nothing removed.")
        return X, 0

    vals = X[num_cols].to_numpy()
    mask_finite = np.all(np.isfinite(vals), axis=1)

    X_clean = X[mask_finite].copy()
    removed_count = len(X) - len(X_clean)

    print(f"Removed {removed_count} rows with NaN / +inf / -inf in numeric columns.")
    return X_clean, removed_count


############### Personal characteristics utils ###############

def add_personal_characteristics(
    df,
    csv_path="PersonalCharacteristicsOfSubjects/personal_characteristics.csv",
):
    """
    Merge participant-level characteristics (height, weight, etc.) into a dataframe.

    Expected columns:
      - df: must contain 'participant_id'
      - CSV: must contain 'Subject' with IDs like F1, M2, ...

    Returns a left-joined dataframe (keeps all rows of df).
    """
    perso = pd.read_csv(csv_path).rename(columns={"Subject": "participant_id"})

    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    perso["participant_id"] = perso["participant_id"].astype(str)

    return df.merge(perso, on="participant_id", how="left")


############### Cross Validation and final fit (single-output) ###############

def cv_select_and_refit_on_full_train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    base_estimator,
    param_grid: Dict[str, Any],
    *,
    model_name: str,
    model_title: str,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    n_outer_splits: int = 5,
    drop_non_numeric: bool = True,
    return_oof: bool = True,
    bundle_dir: str = "runs",
):
    """
    Group-aware hyperparameter selection (outer GroupKFold), then refit on full train.

    Selection criterion:
      - mean MAE over the outer folds.

    Outputs:
      - a saved bundle (.joblib) with model + metadata + CV summary (+ optional OOF).
    """
    print(f"\n[{model_name}] Launching...")

    y = y_train.iloc[:, 0].reset_index(drop=True)
    X = X_train.reset_index(drop=True).copy()

    if participant_col not in X.columns:
        raise ValueError(f"Missing column '{participant_col}' in X_train.")

    groups = X[participant_col]

    # Build feature matrix
    drop_cols = [participant_col]
    if time_col in X.columns:
        drop_cols.append(time_col)

    X_feat = X.drop(columns=drop_cols, errors="ignore")
    if drop_non_numeric:
        X_feat = X_feat.select_dtypes(include=["number", "bool"])

    feature_cols = X_feat.columns.tolist()

    # Outer CV over param combinations
    outer_cv = GroupKFold(n_splits=n_outer_splits)
    combos = list(ParameterGrid(param_grid))
    print(f"[{model_name}] #param combinations: {len(combos)} | outer folds: {n_outer_splits}")

    rows = []
    for ci, params in enumerate(combos, start=1):
        print(f"  Combo {ci}/{len(combos)}: {params}")
        fold_mae = []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_feat, y, groups), start=1):
            est = clone(base_estimator)
            est.set_params(**params)
            est.fit(X_feat.iloc[tr_idx], y.iloc[tr_idx])

            pred = est.predict(X_feat.iloc[te_idx])
            fold_mae.append(mean_absolute_error(y.iloc[te_idx], pred))

        rows.append({
            "params": params,
            "mae_mean_outer": float(np.mean(fold_mae)),
            "mae_std_outer": float(np.std(fold_mae)),
            "mae_per_fold": fold_mae,
        })

        print(
            f"    MAE mean/std over folds: "
            f"{rows[-1]['mae_mean_outer']:.4f} ± {rows[-1]['mae_std_outer']:.4f}"
        )

    summary_df = pd.DataFrame(rows).sort_values("mae_mean_outer").reset_index(drop=True)
    best_params = summary_df.loc[0, "params"]

    print(f"\n[{model_name}] BEST params = {best_params}")
    print(f"[{model_name}] BEST MAE = {summary_df.loc[0, 'mae_mean_outer']:.4f}")

    # Optional OOF predictions
    oof_df = None
    if return_oof:
        oof_pred = pd.Series(index=X.index, dtype=float)
        oof_fold = pd.Series(index=X.index, dtype=int)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_feat, y, groups), start=1):
            est = clone(base_estimator)
            est.set_params(**best_params)
            est.fit(X_feat.iloc[tr_idx], y.iloc[tr_idx])

            oof_pred.iloc[te_idx] = est.predict(X_feat.iloc[te_idx])
            oof_fold.iloc[te_idx] = fold_idx

        oof_df = pd.DataFrame({
            participant_col: X[participant_col],
            "y_true": y,
            "y_pred": oof_pred,
            "fold": oof_fold,
        })
        if time_col in X.columns:
            oof_df[time_col] = X[time_col]

    # Final refit on full train
    final_model = clone(base_estimator)
    final_model.set_params(**best_params)
    final_model.fit(X_feat, y)

    train_meta = {
        "participant_col": participant_col,
        "time_col": time_col,
        "drop_cols": drop_cols,
        "drop_non_numeric": drop_non_numeric,
        "feature_cols": feature_cols,
    }

    return create_and_save_bundle(
        final_model=final_model,
        best_params=best_params,
        train_meta=train_meta,
        summary_df=summary_df,
        oof_df=oof_df,
        model_name=model_name,
        model_title=model_title,
        bundle_dir=bundle_dir,
        extra_info={"n_outer_splits": n_outer_splits},
    )


############### Cross Validation and final fit (multi-output) ###############

def cv_select_and_refit_on_full_train_multioutput(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # DataFrame (n,2): [thigh_flux, back_flux]
    base_estimator,
    param_grid,
    model_name: str = "model",
    model_title: str | None = None,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    n_outer_splits: int = 5,
    drop_non_numeric: bool = True,
    return_oof: bool = True,
):
    """
    Performs cross-validated hyperparameter selection for multi-output regression,
    then refits on the full training set with the best parameters.
    """
    X = X_train.reset_index(drop=True).copy()
    Y = y_train.reset_index(drop=True).copy()

    if not isinstance(Y, pd.DataFrame) or Y.shape[1] != 2:
        raise ValueError("y_train must be a DataFrame with exactly 2 columns (thigh, back).")
    if len(X) != len(Y):
        raise ValueError(f"X_train and y_train must have same length: {len(X)} vs {len(Y)}")
    if participant_col not in X.columns:
        raise ValueError(f"Missing column '{participant_col}' in X_train.")

    y_cols = list(Y.columns)
    groups = X[participant_col].reset_index(drop=True)

    # Build feature matrix
    drop_cols = [participant_col]
    if time_col in X.columns:
        drop_cols.append(time_col)

    X_feat = X.drop(columns=drop_cols, errors="ignore")
    if drop_non_numeric:
        X_feat = X_feat.select_dtypes(include=["number", "bool"]).copy()

    feature_cols = X_feat.columns.tolist()

    # Outer CV over param combinations
    outer_cv = GroupKFold(n_splits=n_outer_splits)
    combos = list(ParameterGrid(param_grid))
    print(f"[{model_name}] #param combinations: {len(combos)} | outer folds: {n_outer_splits}")

    rows = []
    for ci, params in enumerate(combos, start=1):
        print(f"\n[{model_name}] Combo {ci}/{len(combos)}: {params}")

        fold_mae_thigh, fold_mae_back, fold_mae_mean = [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_feat, Y.iloc[:, 0], groups), start=1):
            X_tr, X_te = X_feat.iloc[tr_idx], X_feat.iloc[te_idx]
            Y_tr, Y_te = Y.iloc[tr_idx], Y.iloc[te_idx]

            est = clone(base_estimator)
            est.set_params(**params)
            est.fit(X_tr, Y_tr)

            pred = np.asarray(est.predict(X_te), dtype=float)
            if pred.ndim != 2 or pred.shape[1] != 2:
                raise ValueError(f"predict() must return (n,2). Got shape={pred.shape}")

            mae_thigh = mean_absolute_error(Y_te.iloc[:, 0], pred[:, 0])
            mae_back  = mean_absolute_error(Y_te.iloc[:, 1], pred[:, 1])
            mae_mean  = 0.5 * (mae_thigh + mae_back)

            fold_mae_thigh.append(float(mae_thigh))
            fold_mae_back.append(float(mae_back))
            fold_mae_mean.append(float(mae_mean))

            print(f"    Fold {fold_idx}: thigh={mae_thigh:.3f} | back={mae_back:.3f} | mean={mae_mean:.3f}")

        rows.append({
            "params": params,
            "mae_thigh_mean_outer": float(np.mean(fold_mae_thigh)),
            "mae_back_mean_outer": float(np.mean(fold_mae_back)),
            "mae_mean_outer": float(np.mean(fold_mae_mean)),
            "mae_thigh_std_outer": float(np.std(fold_mae_thigh)),
            "mae_back_std_outer": float(np.std(fold_mae_back)),
            "mae_std_outer": float(np.std(fold_mae_mean)),
            "mae_thigh_per_fold": fold_mae_thigh,
            "mae_back_per_fold": fold_mae_back,
            "mae_mean_per_fold": fold_mae_mean,
        })

        print(
            f"    → MEAN ± STD | "
            f"thigh: {np.mean(fold_mae_thigh):.3f} ± {np.std(fold_mae_thigh):.3f} | "
            f"back: {np.mean(fold_mae_back):.3f} ± {np.std(fold_mae_back):.3f} | "
            f"mean: {np.mean(fold_mae_mean):.3f} ± {np.std(fold_mae_mean):.3f}"
        )

    summary_df = pd.DataFrame(rows).sort_values("mae_mean_outer").reset_index(drop=True)
    best_params = summary_df.loc[0, "params"]

    print(f"\n[{model_name}] BEST params by mean(2 outputs) outer MAE = {summary_df.loc[0, 'mae_mean_outer']:.4f}")
    print("  best_params:", best_params)
    print(
        f"  thigh MAE: {summary_df.loc[0, 'mae_thigh_mean_outer']:.4f} | "
        f"back MAE: {summary_df.loc[0, 'mae_back_mean_outer']:.4f}"
    )

    # Optional OOF predictions
    oof_df = None
    if return_oof:
        oof_pred = np.full((len(X), 2), np.nan, dtype=float)
        oof_fold = np.full((len(X),), -1, dtype=int)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_feat, Y.iloc[:, 0], groups), start=1):
            est = clone(base_estimator)
            est.set_params(**best_params)
            est.fit(X_feat.iloc[tr_idx], Y.iloc[tr_idx])

            pred = np.asarray(est.predict(X_feat.iloc[te_idx]), dtype=float)
            if pred.ndim != 2 or pred.shape[1] != 2:
                raise ValueError(f"OOF predict() must return (n,2). Got shape={pred.shape}")

            oof_pred[te_idx, :] = pred
            oof_fold[te_idx] = fold_idx

        oof_df = pd.DataFrame({
            participant_col: X[participant_col].values,
            "fold": oof_fold,
            f"y_true_{y_cols[0]}": Y.iloc[:, 0].values,
            f"y_true_{y_cols[1]}": Y.iloc[:, 1].values,
            f"y_pred_{y_cols[0]}": oof_pred[:, 0],
            f"y_pred_{y_cols[1]}": oof_pred[:, 1],
        })
        if time_col in X.columns:
            oof_df[time_col] = X[time_col].values

    # Final refit on full train
    final_model = clone(base_estimator)
    final_model.set_params(**best_params)
    final_model.fit(X_feat, Y)

    train_meta = {
        "participant_col": participant_col,
        "time_col": time_col,
        "drop_cols": drop_cols,
        "drop_non_numeric": drop_non_numeric,
        "feature_cols": feature_cols,
        "y_cols": y_cols,
        "model_title": model_title,
    }

    return create_and_save_bundle(
        final_model=final_model,
        best_params=best_params,
        train_meta=train_meta,
        summary_df=summary_df,
        oof_df=oof_df,
        model_name=model_name,
        model_title=model_title,
        bundle_dir="runs",
        extra_info={"n_outer_splits": n_outer_splits},
    )


############### Bundle saving utils ###############

def create_and_save_bundle(
    *,
    final_model,
    best_params: Dict[str, Any],
    train_meta: Dict[str, Any],
    summary_df: pd.DataFrame,
    oof_df: Optional[pd.DataFrame],
    model_name: str,
    model_title: str,
    bundle_dir: str | Path = "runs",
    add_timestamp: bool = True,
    extra_info: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Save everything needed to reproduce a run in a single .joblib bundle.
    Bundle contains:
      - model (fitted)
      - best_params
      - train_meta (features, dropped columns, etc.)
      - summary_df (CV results)
      - oof_df (optional OOF predictions)
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{model_title}"
    bundle_path = bundle_dir / f"{tag}.joblib"

    info = {
        "model_name": model_name,
        "model_title": model_title,
        "tag": tag,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra_info:
        info.update(extra_info)

    bundle = {
        "model": final_model,
        "best_params": best_params,
        "train_meta": train_meta,
        "summary_df": summary_df,
        "oof_df": oof_df,
        "info": info,
        "bundle_path": str(bundle_path),
    }

    joblib.dump(bundle, bundle_path)
    print(f"\n[SAVE] bundle -> {bundle_path}")
    return bundle


############### Plotting helpers (single-output) ###############

def _make_sessions(time_series: pd.Series, gap_minutes: float) -> pd.Series:
    """Build session_id based on time gaps larger than `gap_minutes`."""
    t = pd.to_datetime(time_series, errors="coerce")
    dt = t.diff()
    gap = pd.to_timedelta(gap_minutes, unit="min")
    return ((dt > gap).fillna(False)).cumsum().astype(int)


def _compress_time(t: pd.Series, session_id: pd.Series) -> pd.Series:
    """
    Create a compressed time axis by removing gaps between sessions.
    Returns datetime-like values so matplotlib formats them nicely.
    """
    t = pd.to_datetime(t, errors="coerce")
    if t.isna().any():
        raise ValueError("Some Time values are not convertible to datetime.")

    t0 = t.iloc[0]
    rel = (t - t0).to_numpy()
    sess = session_id.to_numpy()

    starts = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
    ends = np.r_[starts[1:] - 1, len(sess) - 1]

    gaps = []
    for i in range(len(starts) - 1):
        gaps.append(rel[starts[i + 1]] - rel[ends[i]])

    cum_gap = np.zeros(len(sess), dtype="timedelta64[ns]")
    total = np.timedelta64(0, "ns")
    for i in range(len(starts) - 1):
        total = total + gaps[i]
        cum_gap[starts[i + 1]:] = total

    rel_compressed = rel - cum_gap
    return pd.to_datetime(t0) + pd.to_timedelta(rel_compressed)


def _insert_breaks_for_plot(df: pd.DataFrame, x_col: str, session_col: str, cols_to_break):
    """Insert NaN rows between sessions so line plots do not connect sessions."""
    parts = []
    for sid, g in df.groupby(session_col, sort=True):
        parts.append(g)

        br = {c: np.nan for c in cols_to_break}
        br[x_col] = pd.NaT
        br[session_col] = sid
        parts.append(pd.DataFrame([br]))

    return pd.concat(parts, ignore_index=True)


def predict_and_plot(
    model,
    X_test: pd.DataFrame,
    y_test,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    drop_cols_for_model: tuple = ("participant_id", "Time"),
    participant_id='F7',
    gap_minutes: float = 60,
    mode: str = "all",          # "all" or "selected"
    session_number: int = 1,    # used if mode="selected"
    compress_gaps: bool = True,
    figsize=(18, 5),
    title=None,
):
    """
    Predict + plot truth vs prediction for one participant.

    - Global MAE is computed on the whole test set.
    - mode="all": plot all sessions for that participant (with optional gap compression).
    - mode="selected": plot a single session based on session_number.
    """
    X = X_test.copy()

    if time_col not in X.columns or participant_col not in X.columns:
        raise ValueError(f"X_test must contain '{participant_col}' and '{time_col}'.")

    if isinstance(y_test, pd.DataFrame):
        y = y_test.iloc[:, 0].to_numpy()
    elif isinstance(y_test, pd.Series):
        y = y_test.to_numpy()
    else:
        y = np.asarray(y_test)

    drop_cols = [c for c in drop_cols_for_model if c in X.columns]
    X_model = X.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number", "bool"])

    y_pred = model.predict(X_model)

    pred_df = pd.DataFrame({
        participant_col: X[participant_col].values,
        time_col: pd.to_datetime(X[time_col], errors="coerce"),
        "y_true": y,
        "y_pred": y_pred,
    })

    if pred_df[time_col].isna().any():
        raise ValueError("Time contains non-convertible values. Clean/parse Time before plotting.")

    global_mae = mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])

    if participant_id is None:
        participant_id = pred_df[participant_col].iloc[0]

    dfp = pred_df[pred_df[participant_col] == participant_id].sort_values(time_col).reset_index(drop=True)
    dfp["session_id"] = _make_sessions(dfp[time_col], gap_minutes=gap_minutes)

    if mode == "all":
        df_plot = dfp.copy()
        session_label = f"all sessions (gap>{gap_minutes}min)"
    elif mode == "selected":
        session_ids_sorted = sorted(dfp["session_id"].unique().tolist())
        if session_number < 1 or session_number > len(session_ids_sorted):
            raise ValueError(f"Invalid session_number={session_number}. This participant has {len(session_ids_sorted)} session(s).")
        chosen_id = session_ids_sorted[session_number - 1]
        df_plot = dfp[dfp["session_id"] == chosen_id].copy()
        session_label = f"selected session #{session_number} (id={chosen_id})"
    else:
        raise ValueError("mode must be 'all' or 'selected'.")

    if compress_gaps and mode == "all":
        df_plot["time_plot"] = _compress_time(df_plot[time_col], df_plot["session_id"])
        x_col = "time_plot"
        x_is_real = False
    else:
        df_plot["time_plot"] = df_plot[time_col]
        x_col = "time_plot"
        x_is_real = True

    df_plot = _insert_breaks_for_plot(df_plot, x_col=x_col, session_col="session_id", cols_to_break=["y_true", "y_pred"])

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(df_plot[x_col], df_plot["y_true"], label="truth", linewidth=1.8)
    ax.plot(df_plot[x_col], df_plot["y_pred"], label="pred", linewidth=1.8)

    y_min = min(df_plot["y_true"].min(), df_plot["y_pred"].min())
    y_max = max(df_plot["y_true"].max(), df_plot["y_pred"].max())
    pad = 0.3 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_ylabel("Target")
    ax.set_xlabel("Time" if x_is_real else "Time (gaps removed)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    tmin = pd.to_datetime(dfp[time_col].min())
    tmax = pd.to_datetime(dfp[time_col].max())
    ax.set_title(
        f"{title if title is not None else 'Result'} | Global test MAE = {global_mae:.2f}\n"
        f"{session_label} | {tmin} → {tmax}"
    )

    plt.tight_layout()
    plt.show()
    return pred_df


############### Plotting utils (multi-output) ###############

def _extract_fitted_estimator(bundle):
    """
    Accepts either:
      - a fitted estimator, or
      - a dict bundle containing a fitted estimator under common keys.
    """
    if bundle is None:
        raise ValueError("bundle is None")
    if hasattr(bundle, "predict"):
        return bundle
    if isinstance(bundle, dict):
        for k in ["best_estimator", "best_model", "estimator", "model", "refit_estimator", "final_estimator"]:
            if k in bundle and hasattr(bundle[k], "predict"):
                return bundle[k]
    raise ValueError("Could not extract a fitted estimator from `bundle`.")


def add_session_from_time_gaps(
    df: pd.DataFrame,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    gap_minutes: float = 10.0,
    session_col_out: str = "session_idx",
) -> pd.DataFrame:
    """Add per-participant session indices based on time gaps > gap_minutes."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values([participant_col, time_col])

    dt = out.groupby(participant_col)[time_col].diff()
    gap = dt > pd.Timedelta(minutes=gap_minutes)

    out[session_col_out] = gap.groupby(out[participant_col]).cumsum().astype(int) + 1
    return out


def plot_and_predict_multioutput_by_gap_session(
    bundle,
    X_df: pd.DataFrame,
    y_df: pd.DataFrame | None = None,
    participant_id: str = "F1",
    session_idx: int = 1,
    gap_minutes: float = 10.0,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    target_cols: tuple[str, str] = ("thigh_flux", "back_flux"),
    drop_cols_for_X: list[str] | None = None,
    drop_non_numeric: bool = True,
    n_points: int | None = None,
):
    """
    Plot predictions for one (participant, session) for a multi-output model.
    Optionally overlays ground truth and returns local MAEs.
    """
    est = _extract_fitted_estimator(bundle)

    X_with_sess = add_session_from_time_gaps(
        X_df,
        participant_col=participant_col,
        time_col=time_col,
        gap_minutes=gap_minutes,
        session_col_out="session_idx",
    )

    mask = (X_with_sess[participant_col] == participant_id) & (X_with_sess["session_idx"] == session_idx)
    X_sub = X_with_sess.loc[mask].copy()

    if X_sub.empty:
        avail = X_with_sess.loc[X_with_sess[participant_col] == participant_id, "session_idx"].unique()
        avail = np.sort(avail) if len(avail) else avail
        raise ValueError(
            f"No rows found for participant_id={participant_id} session_idx={session_idx}. "
            f"Available session_idx: {avail}"
        )

    X_sub[time_col] = pd.to_datetime(X_sub[time_col], errors="coerce")
    X_sub = X_sub.sort_values(time_col)

    selected_index = X_sub.index.copy()

    if n_points is not None and len(X_sub) > n_points:
        idx = np.linspace(0, len(X_sub) - 1, n_points).astype(int)
        X_sub = X_sub.iloc[idx].copy()
        selected_index = X_sub.index.copy()

    X_pred = X_sub.drop(columns=["session_idx"], errors="ignore").copy()
    if drop_cols_for_X is not None:
        X_pred = X_pred.drop(columns=drop_cols_for_X, errors="ignore")
    if drop_non_numeric:
        X_pred = X_pred.select_dtypes(include=[np.number])

    y_pred = np.asarray(est.predict(X_pred))
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError(f"Expected predictions shape (n,2), got {y_pred.shape}")

    out = pd.DataFrame({
        time_col: X_sub[time_col].values,
        "y_pred_thigh": y_pred[:, 0],
        "y_pred_back":  y_pred[:, 1],
    }, index=selected_index)

    mae_thigh = None
    mae_back = None

    if y_df is not None:
        if selected_index.isin(y_df.index).all():
            y_sub = y_df.loc[selected_index]
        else:
            pos = X_df.index.get_indexer(selected_index)
            if (pos < 0).any():
                raise ValueError("Some selected rows cannot be mapped back to X_df positions.")
            y_sub = y_df.iloc[pos]

        out["y_true_thigh"] = y_sub[target_cols[0]].values
        out["y_true_back"]  = y_sub[target_cols[1]].values

        mae_thigh = mean_absolute_error(out["y_true_thigh"], out["y_pred_thigh"])
        mae_back  = mean_absolute_error(out["y_true_back"],  out["y_pred_back"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    title_thigh = f"{participant_id} | session_idx={session_idx} (gap>{gap_minutes}min) — Thigh"
    title_back  = f"{participant_id} | session_idx={session_idx} (gap>{gap_minutes}min) — Back"
    if mae_thigh is not None:
        title_thigh += f" | local MAE={mae_thigh:.2f}"
    if mae_back is not None:
        title_back += f" | local MAE={mae_back:.2f}"

    axes[0].plot(out[time_col], out["y_pred_thigh"], label="Pred thigh")
    if "y_true_thigh" in out.columns:
        axes[0].plot(out[time_col], out["y_true_thigh"], label="True thigh", alpha=0.7)
    axes[0].set_title(title_thigh)
    axes[0].set_ylabel("thigh_flux")
    axes[0].legend()

    axes[1].plot(out[time_col], out["y_pred_back"], label="Pred back")
    if "y_true_back" in out.columns:
        axes[1].plot(out[time_col], out["y_true_back"], label="True back", alpha=0.7)
    axes[1].set_title(title_back)
    axes[1].set_ylabel("back_flux")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return out, mae_thigh, mae_back


############### Classification plots ###############

def plot_confusion_matrix_and_scores(y_val, y_pred):
    """
    Plot a row-normalized confusion matrix with percentages + raw counts,
    and display key metrics (Accuracy/Precision/Recall/F1).
    """
    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)

    cm = confusion_matrix(y_val, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    class_names = ["Standing", "Seated"]

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n(n={cm[i, j]})"

    plt.figure(figsize=(6.5, 6))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[f"Pred: {c}" for c in class_names],
        yticklabels=[f"True: {c}" for c in class_names],
        linewidths=0.5,
    )

    plt.title("Normalized Confusion Matrix\n(% per class + counts per cell)", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")

    plt.figtext(
        0.5,
        -0.12,
        f"Accuracy = {acc:.3f}    |    Precision = {prec:.3f}    |    Recall = {rec:.3f}    |    F1-score = {f1:.3f}",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()
