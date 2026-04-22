import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Регрессия по 1/T", layout="wide")
st.title("Регрессия по 1/T")

COLUMN_RENAMES = {
    "dэкв_мкм": "d_equiv_um",
    "tau_h": "tau_h",
    "T_C": "T_C",
    "G": "G",
    "c_sigma_pct": "c_sigma_pct",
}

REQUIRED_COLUMNS = ["G", "T_C", "tau_h", "d_equiv_um"]
OPTIONAL_COLUMNS = ["c_sigma_pct"]


def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    if uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df = df.rename(columns=COLUMN_RENAMES)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"В файле не хватает обязательных колонок: {', '.join(missing)}")

    if "c_sigma_pct" not in df.columns:
        df["c_sigma_pct"] = 0.1

    keep_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = df[keep_cols].copy()

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[(df["tau_h"] > 0) & (df["d_equiv_um"] > 0) & (df["T_C"] > 0)]
    df["T_K"] = df["T_C"] + 273.15
    df["c_sigma_pct"] = df["c_sigma_pct"].replace(0, 0.1)
    return df


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def compute_temp_metrics(t_true, t_pred):
    mask = np.isfinite(t_pred)
    if not mask.any():
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
    y_true = t_true[mask]
    y_pred = t_pred[mask]
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def fit_linear_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = beta[0]
    coef = beta[1:]
    y_pred = X_design @ beta
    return intercept, coef, y_pred


def fit_inverse_temp_model(df, include_G=True):
    c_sigma = df["c_sigma_pct"].replace(0, 0.1)
    if include_G:
        features = np.column_stack([
            np.log(df["d_equiv_um"]),
            np.log(df["tau_h"]),
            df["G"],
            np.log(c_sigma),
        ])
    else:
        features = np.column_stack([
            np.log(df["d_equiv_um"]),
            np.log(df["tau_h"]),
            np.log(c_sigma),
        ])

    target = 1.0 / df["T_K"].values
    intercept, coef, y_pred = fit_linear_regression(features, target)

    with np.errstate(divide="ignore"):
        t_pred_k = np.where(y_pred > 0, 1.0 / y_pred, np.nan)

    metrics = compute_temp_metrics(df["T_C"].values, t_pred_k - 273.15)

    return {
        "intercept": intercept,
        "coef": coef,
        "t_pred_k": t_pred_k,
        "metrics": metrics,
        "include_G": include_G,
    }


def format_formula(model_data):
    coef = model_data["coef"]
    intercept = model_data["intercept"]
    if model_data["include_G"]:
        return (
            f"1/T = {intercept:.8f} + ({coef[0]:.8f})·ln(D) + ({coef[1]:.8f})·ln(τ) + "
            f"({coef[2]:.8f})·G + ({coef[3]:.8f})·ln(cσ)"
        )
    return (
        f"1/T = {intercept:.8f} + ({coef[0]:.8f})·ln(D) + ({coef[1]:.8f})·ln(τ) + "
        f"({coef[2]:.8f})·ln(cσ)"
    )


def calculate_point_influence(df, include_G=True):
    base_model = fit_inverse_temp_model(df, include_G=include_G)
    base_rmse = base_model["metrics"]["rmse"]
    rows = []

    if len(df) <= (4 if include_G else 3):
        return pd.DataFrame()

    for idx in df.index:
        reduced = df.drop(index=idx)
        if len(reduced) < 3:
            continue
        try:
            reduced_include_g = include_G and reduced["G"].nunique() > 1
            reduced_model = fit_inverse_temp_model(reduced, include_G=reduced_include_g)
            reduced_rmse = reduced_model["metrics"]["rmse"]
            delta_rmse = base_rmse - reduced_rmse
            rows.append({
                "index": int(idx),
                "G": df.loc[idx, "G"],
                "T_C": df.loc[idx, "T_C"],
                "tau_h": df.loc[idx, "tau_h"],
                "d_equiv_um": df.loc[idx, "d_equiv_um"],
                "c_sigma_pct": df.loc[idx, "c_sigma_pct"],
                "RMSE_без_точки": reduced_rmse,
                "Ухудшение_RMSE": delta_rmse,
            })
        except Exception:
            continue

    influence_df = pd.DataFrame(rows)
    if not influence_df.empty:
        influence_df = influence_df.sort_values("Ухудшение_RMSE", ascending=False)
    return influence_df


def render_model_block(df, title):
    st.subheader(title)

    include_G = df["G"].nunique() > 1
    model_data = fit_inverse_temp_model(df, include_G=include_G)
    formula = format_formula(model_data)
    coef = model_data["coef"]
    intercept = model_data["intercept"]

    st.markdown("### Коэффициенты модели")
    coef_rows = [{"Параметр": "intercept", "Значение": intercept}]
    coef_rows.append({"Параметр": "ln(D)", "Значение": coef[0]})
    coef_rows.append({"Параметр": "ln(τ)", "Значение": coef[1]})
    if include_G:
        coef_rows.append({"Параметр": "G", "Значение": coef[2]})
        coef_rows.append({"Параметр": "ln(cσ)", "Значение": coef[3]})
    else:
        coef_rows.append({"Параметр": "ln(cσ)", "Значение": coef[2]})
    st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)

    st.markdown("### Формула модели")
    st.code(formula)

    st.markdown("### Качество модели")
    quality_df = pd.DataFrame([
        {
            "RMSE, °C": model_data["metrics"]["rmse"],
            "MAE, °C": model_data["metrics"]["mae"],
            "R²": model_data["metrics"]["r2"],
            "Число точек": len(df),
            "G учитывается": "Да" if include_G else "Нет",
        }
    ])
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.markdown("### Сравнение факт / прогноз")
    pred_df = df.copy()
    pred_df["T_pred_C"] = model_data["t_pred_k"] - 273.15
    pred_df["Отклонение_°C"] = pred_df["T_pred_C"] - pred_df["T_C"]
    st.dataframe(
        pred_df[["G", "T_C", "T_pred_C", "Отклонение_°C", "tau_h", "d_equiv_um", "c_sigma_pct"]],
        use_container_width=True,
        hide_index=True,
    )

    chart_df = pred_df[["T_C", "T_pred_C"]].copy().reset_index(drop=True)
    st.markdown("### График факт / прогноз")
    st.line_chart(chart_df)

    st.markdown("### Влияние каждой точки на качество модели")
    influence_df = calculate_point_influence(df, include_G=include_G)
    if influence_df.empty:
        st.info("Недостаточно данных, чтобы устойчиво оценить влияние каждой точки.")
    else:
        st.caption("Положительное значение в столбце «Ухудшение_RMSE» означает, что точка ухудшает модель: без неё RMSE становится меньше.")
        st.dataframe(influence_df, use_container_width=True, hide_index=True)


uploaded_file = st.sidebar.file_uploader("Загрузите Excel/CSV в формате model_sigma", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("Загрузите файл Excel или CSV в том же формате, что использует model_sigma.")
else:
    try:
        df = load_data(uploaded_file)
        if df is None or df.empty:
            st.warning("В файле нет корректных данных для анализа.")
        else:
            st.success(f"Загружено точек: {len(df)}")

            tab_all, tab_by_grain = st.tabs(["Общая модель", "Модели по номеру зерна"])

            with tab_all:
                render_model_block(df, "Модель по всем данным")

            with tab_by_grain:
                grain_values = sorted(df["G"].dropna().unique())
                if len(grain_values) == 0:
                    st.info("Нет доступных номеров зерна.")
                else:
                    selected_g = st.selectbox("Выберите номер зерна", grain_values)
                    df_g = df[df["G"] == selected_g].copy()
                    if df_g.empty:
                        st.info("Для выбранного номера зерна нет данных.")
                    else:
                        render_model_block(df_g, f"Модель для зерна G = {selected_g}")
    except Exception as e:
        st.error(f"Ошибка: {e}")
