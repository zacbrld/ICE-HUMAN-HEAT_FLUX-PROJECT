import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


def load_lstm_bundle(bundle_path: str, LSTMRegressorClass, device=None):
    # device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(bundle_path, map_location=device)

    # rebuild scaler
    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # metadata
    num_cols  = ckpt["num_cols"]
    window    = int(ckpt["window"])
    input_dim = int(ckpt["input_dim"])

    # model
    model = LSTMRegressorClass(
        input_dim=input_dim,
        hidden_dim=int(ckpt["hidden_dim"]),
        fc_dim=int(ckpt["fc_dim"]),
        output_dim=int(ckpt["output_dim"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded model from bundle.")
    print("window =", window, "| hidden_dim =", ckpt["hidden_dim"], "| n_features =", len(num_cols))

    return {
        "ckpt": ckpt,
        "model": model,
        "scaler": scaler,
        "num_cols": num_cols,
        "window": window,
        "input_dim": input_dim,
        "device": device,
    }


def align_and_scale_X_test(X_test, num_cols, scaler,
                           id_cols=("participant_id", "Time", "Session")):
    # checks
    for c in id_cols:
        if c not in X_test.columns:
            raise ValueError(f"Colonne '{c}' manquante dans X_test.")

    missing = [c for c in num_cols if c not in X_test.columns]
    if missing:
        raise ValueError(f"Colonnes features manquantes dans X_test: {missing}")

    # align (ordre exact)
    X_test_aligned = X_test[list(id_cols) + list(num_cols)].copy()

    # scale features only
    X_test_scaled = X_test_aligned.copy()
    X_test_scaled[list(num_cols)] = scaler.transform(X_test_scaled[list(num_cols)])

    return X_test_scaled, X_test_aligned

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


def predict_and_plot(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    participant_col: str = "participant_id",
    time_col: str = "Time",
    participant_id=None,
    gap_minutes: float = 60,
    mode: str = "all",
    session_number: int = 1,
    compress_gaps: bool = True,
    figsize=(18, 5),
    title=None,
    window: int = 5,
    target: str = "back_flux",
    device=None,
    batch_size: int = 1024,
):
    X = X_test.copy()
    X[participant_col] = X[participant_col].astype(str)
    if participant_id is not None:
        participant_id = str(participant_id)

    if time_col not in X.columns or participant_col not in X.columns:
        raise ValueError(f"X_test doit contenir '{participant_col}' et '{time_col}'.")

    if not isinstance(y_test, pd.DataFrame):
        raise ValueError("y_test doit être un pd.DataFrame (avec back_flux et thigh_flux).")

    if target not in y_test.columns:
        raise ValueError(f"target='{target}' introuvable dans y_test. Colonnes disponibles: {list(y_test.columns)}")

    if device is None:
        device = next(model.parameters()).device

    # num_cols doit exister globalement (on le fixe après load bundle)
    feature_cols = num_cols
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans X_test : {missing}")

    leak = [c for c in ["back_flux", "thigh_flux"] if c in feature_cols]
    if leak:
        raise ValueError(f"Fuite: target(s) présente(s) dans les features: {leak}")

    t_idx = 0 if target == "back_flux" else 1

    rows = []
    model.eval()

    with torch.no_grad():
        for pid, grp in X[[participant_col, time_col] + feature_cols].groupby(participant_col):
            grp = grp.sort_values(time_col)
            idx = grp.index.to_numpy()

            y_grp = y_test.loc[idx, target].to_numpy()

            if len(idx) <= window:
                continue

            X_np = grp[feature_cols].to_numpy(dtype=np.float32)

            n_seq = len(idx) - window
            times_target = pd.to_datetime(grp[time_col].iloc[window:].values, errors="coerce")
            y_true_target = y_grp[window:]

            preds_out = np.zeros(n_seq, dtype=np.float32)

            for start in range(0, n_seq, batch_size):
                end = min(start + batch_size, n_seq)

                batch = np.stack([X_np[i:i+window] for i in range(start, end)], axis=0)
                xb = torch.tensor(batch, dtype=torch.float32, device=device)
                pred2 = model(xb).detach().cpu().numpy()  # (B,2)
                preds_out[start:end] = pred2[:, t_idx]

            for t, yt, yp in zip(times_target, y_true_target, preds_out):
                rows.append((pid, t, float(yt), float(yp)))

    pred_df = pd.DataFrame(rows, columns=[participant_col, time_col, "y_true", "y_pred"])
    if pred_df.empty:
        raise ValueError("Aucune séquence créée pour le plot. Vérifie window et la longueur des séries.")

    if pred_df[time_col].isna().any():
        raise ValueError("Time contient des valeurs non convertibles en datetime.")

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
        if session_number > len(session_ids_sorted):
            raise ValueError(f"session_number={session_number} invalide: {len(session_ids_sorted)} session(s).")
        chosen_id = session_ids_sorted[session_number - 1]
        df_plot = dfp[dfp["session_id"] == chosen_id].copy()
        session_label = f"selected session #{session_number} (id={chosen_id})"
    else:
        raise ValueError("mode doit être 'all' ou 'selected'.")

    if compress_gaps and mode == "all":
        df_plot["time_plot"] = _compress_time(df_plot[time_col], df_plot["session_id"])
        x_is_real = False
    else:
        df_plot["time_plot"] = df_plot[time_col]
        x_is_real = True

    df_plot = _insert_breaks_for_plot(
        df_plot,
        x_col="time_plot",
        session_col="session_id",
        cols_to_break=["y_true", "y_pred"]
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(df_plot["time_plot"], df_plot["y_true"], label="truth", linewidth=1.8)
    ax.plot(df_plot["time_plot"], df_plot["y_pred"], label="pred",  linewidth=1.8)

    ax.set_ylabel(target)
    ax.set_xlabel("Time" if x_is_real else "Time (gaps removed)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    tmin = pd.to_datetime(dfp[time_col].min())
    tmax = pd.to_datetime(dfp[time_col].max())
    ax.set_title(
        f"{title if title is not None else 'Result'} | Global test MAE ({target}) = {global_mae:.2f}\n"
        f"{session_label} | {tmin} → {tmax}"
    )

    plt.tight_layout()
    plt.show()

    return pred_df

def _make_sessions(time_series: pd.Series, gap_minutes: float) -> pd.Series:
    t = pd.to_datetime(time_series, errors="coerce")
    dt = t.diff()
    gap = pd.to_timedelta(gap_minutes, unit="min")
    new_session = (dt > gap).fillna(False)
    return new_session.cumsum().astype(int)


def _compress_time(t: pd.Series, session_id: pd.Series) -> pd.Series:
    t = pd.to_datetime(t, errors="coerce")
    if t.isna().any():
        raise ValueError("Certaines valeurs de Time ne sont pas convertibles en datetime.")

    # Temps relatif (timedelta) depuis le tout premier point
    t0 = t.iloc[0]
    rel = t - t0

    # Calcule les "gaps" entre fin de session i et début de session i+1
    # et les soustrait cumulativement
    sess = session_id.to_numpy()
    rel_np = rel.to_numpy()

    # Indices de début de session
    starts = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
    # Indices de fin de session
    ends = np.r_[starts[1:] - 1, len(sess) - 1]

    gaps = []
    for i in range(len(starts) - 1):
        end_i = ends[i]
        start_next = starts[i + 1]
        gap_i = rel_np[start_next] - rel_np[end_i]
        gaps.append(gap_i)
    # Cumulative gaps
    cum_gap = np.zeros(len(sess), dtype="timedelta64[ns]")
    total = np.timedelta64(0, "ns")
    for i in range(len(starts) - 1):
        end_i = ends[i]
        start_next = starts[i + 1]
        total = total + gaps[i]
        cum_gap[start_next:] = total

    rel_compressed = rel_np - cum_gap
    # Retransforme en datetime (ancré à t0)
    return pd.to_datetime(t0) + pd.to_timedelta(rel_compressed)


def _insert_breaks_for_plot(df: pd.DataFrame, x_col: str, session_col: str, cols_to_break):
    parts = []
    for sid, g in df.groupby(session_col, sort=True):
        parts.append(g)
        # ligne "break"
        br = {c: np.nan for c in cols_to_break}
        br[x_col] = pd.NaT
        br[session_col] = sid
        parts.append(pd.DataFrame([br]))
    out = pd.concat(parts, ignore_index=True)
    return out
