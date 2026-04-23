import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Регрессия по 1/T", layout="wide")
st.title("Регрессия по 1/T")

COLUMN_RENAMES = {
    "dэкв_мкм": "d_equiv_um",
    "tau_h": "tau_h",
    "T_C": "T_C",
    "G": "G",
    "c_sigma_pct": "c_sigma_pct",
}

SIGMA_TEMP_MIN = 560
SIGMA_TEMP_MAX = 900

DEMO_ROWS = [
    {"G": 3, "T_C": 580, "tau_h": 1200, "d_equiv_um": 0.42, "c_sigma_pct": 1.2},
    {"G": 3, "T_C": 600, "tau_h": 2400, "d_equiv_um": 0.71, "c_sigma_pct": 2.4},
    {"G": 4, "T_C": 620, "tau_h": 3600, "d_equiv_um": 1.05, "c_sigma_pct": 3.8},
    {"G": 4, "T_C": 640, "tau_h": 4800, "d_equiv_um": 1.42, "c_sigma_pct": 5.0},
    {"G": 5, "T_C": 660, "tau_h": 6000, "d_equiv_um": 1.86, "c_sigma_pct": 6.7},
    {"G": 5, "T_C": 680, "tau_h": 7200, "d_equiv_um": 2.25, "c_sigma_pct": 8.3},
    {"G": 6, "T_C": 700, "tau_h": 8400, "d_equiv_um": 2.71, "c_sigma_pct": 10.1},
    {"G": 6, "T_C": 720, "tau_h": 9600, "d_equiv_um": 3.08, "c_sigma_pct": 11.4},
]


def load_data(uploaded):
    if uploaded is None:
        return pd.DataFrame(), 0, 0
    if uploaded.name.endswith(".xlsx") or uploaded.name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    df = df.rename(columns=COLUMN_RENAMES)
    expected = set(COLUMN_RENAMES.values())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"В файле не хватает колонок: {', '.join(missing)}")
    df = df[list(expected)].copy()
    for col in ["G", "T_C", "tau_h", "d_equiv_um", "c_sigma_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["G", "T_C", "tau_h", "d_equiv_um"])
    df["T_K"] = df["T_C"] + 273.15
    df["c_sigma_pct"] = df["c_sigma_pct"].replace(0, 0.1)
    total_rows = len(df)
    mask_valid = (df["G"] > 0) & (df["T_C"] >= SIGMA_TEMP_MIN) & (df["T_C"] <= SIGMA_TEMP_MAX)
    filtered_out = total_rows - mask_valid.sum()
    df = df[mask_valid]
    return df, total_rows, filtered_out


def load_demo_data():
    df = pd.DataFrame(DEMO_ROWS)
    df["T_K"] = df["T_C"] + 273.15
    return df


def compute_temp_metrics(T_true, T_pred):
    mask = np.isfinite(T_pred)
    if not mask.any():
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
    rmse = math.sqrt(mean_squared_error(T_true[mask], T_pred[mask]))
    r2 = r2_score(T_true[mask], T_pred[mask])
    mae = np.mean(np.abs(T_true[mask] - T_pred[mask]))
    return {"rmse": rmse, "r2": r2, "mae": mae}


def fit_inverse_temp_model(df, include_G=True):
    df = df.copy()
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
    model = LinearRegression().fit(features, 1.0 / df["T_K"])
    y_pred = model.predict(features)
    with np.errstate(divide="ignore"):
        T_pred = np.where(y_pred > 0, 1.0 / y_pred, np.nan)
    return {
        "model": model,
        "T_pred_K": T_pred,
        "metrics": compute_temp_metrics(df["T_C"].values, T_pred - 273.15),
        "include_G": include_G,
    }


def format_formula(model_data):
    coef = model_data["model"].coef_
    intercept = model_data["model"].intercept_
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
    min_points = 5 if include_G else 4
    if len(df) < min_points:
        return pd.DataFrame()
    for idx in df.index:
        reduced = df.drop(index=idx)
        if len(reduced) < 3:
            continue
        try:
            reduced_include_g = include_G and reduced["G"].nunique() > 1
            reduced_model = fit_inverse_temp_model(reduced, include_G=reduced_include_g)
            reduced_rmse = reduced_model["metrics"]["rmse"]
            rows.append({
                "index": int(idx),
                "G": df.loc[idx, "G"],
                "T_C": df.loc[idx, "T_C"],
                "tau_h": df.loc[idx, "tau_h"],
                "d_equiv_um": df.loc[idx, "d_equiv_um"],
                "c_sigma_pct": df.loc[idx, "c_sigma_pct"],
                "RMSE без точки": reduced_rmse,
                "Ухудшение качества": base_rmse - reduced_rmse,
            })
        except Exception:
            continue
    influence_df = pd.DataFrame(rows)
    if not influence_df.empty:
        influence_df = influence_df.sort_values("Ухудшение качества", ascending=False)
    return influence_df


def render_model_block(df, title):
    st.subheader(title)
    include_G = df["G"].nunique() > 1
    model_data = fit_inverse_temp_model(df, include_G=include_G)
    formula = format_formula(model_data)
    coef = model_data["model"].coef_
    intercept = model_data["model"].intercept_

    st.markdown("### Коэффициенты модели")
    rows = [{"Параметр": "intercept", "Значение": intercept}]
    rows.append({"Параметр": "ln(D)", "Значение": coef[0]})
    rows.append({"Параметр": "ln(τ)", "Значение": coef[1]})
    if include_G:
        rows.append({"Параметр": "G", "Значение": coef[2]})
        rows.append({"Параметр": "ln(cσ)", "Значение": coef[3]})
    else:
        rows.append({"Параметр": "ln(cσ)", "Значение": coef[2]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Формула модели")
    st.code(formula)

    st.markdown("### Качество модели")
    quality_df = pd.DataFrame([
        {
            "RMSE, °C": model_data["metrics"]["rmse"],
            "MAE, °C": model_data["metrics"]["mae"],
            "R²": model_data["metrics"]["r2"],
            "Количество точек": len(df),
            "G в модели": "Да" if include_G else "Нет",
        }
    ])
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    pred_df = df.copy()
    pred_df["T_pred_C"] = model_data["T_pred_K"] - 273.15
    pred_df["Отклонение_°C"] = pred_df["T_pred_C"] - pred_df["T_C"]

    st.markdown("### Прогноз по точкам")
    st.dataframe(
        pred_df[["G", "T_C", "T_pred_C", "Отклонение_°C", "tau_h", "d_equiv_um", "c_sigma_pct"]],
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.isfinite(pred_df["T_pred_C"])
    ax.scatter(pred_df.loc[mask, "T_C"], pred_df.loc[mask, "T_pred_C"], color="tab:blue", alpha=0.7)
    min_v = min(pred_df.loc[mask, "T_C"].min(), pred_df.loc[mask, "T_pred_C"].min()) if mask.any() else 0
    max_v = max(pred_df.loc[mask, "T_C"].max(), pred_df.loc[mask, "T_pred_C"].max()) if mask.any() else 1
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
    ax.set_xlabel("Наблюдаемая T, °C")
    ax.set_ylabel("Предсказанная T, °C")
    ax.set_title("Факт vs прогноз")
    st.pyplot(fig)

    st.markdown("### Влияние каждой точки на качество модели")
    influence_df = calculate_point_influence(df, include_G=include_G)
    if influence_df.empty:
        st.info("Недостаточно данных для оценки влияния точек.")
    else:
        st.caption("Если удаление точки уменьшает RMSE, значит эта точка ухудшает качество модели.")
        st.dataframe(influence_df, use_container_width=True, hide_index=True)


def main():
    st.markdown(
        "Отдельная программа только для модели **регрессии по 1/T**. "
        "Берёт данные в том же формате, что и model_sigma."
    )

    with st.sidebar.expander("Загрузка данных"):
        uploaded = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx", "xls"])
        use_demo = st.toggle("Показать demo-данные", value=True, disabled=uploaded is not None)
        st.caption("Обязательные колонки: G, T_C, tau_h, dэкв_мкм; c_sigma_pct — опционально")

    if uploaded is None:
        if not use_demo:
            st.info("Загрузите файл с данными или включите demo-режим слева.")
            return
        df = load_demo_data()
        st.info("Сейчас показаны demo-данные.")
    else:
        try:
            df, total_rows, filtered_out = load_data(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return
        if filtered_out > 0:
            st.warning(
                f"{filtered_out} из {total_rows} строк не вошли в анализ, потому что находятся вне диапазона {SIGMA_TEMP_MIN}–{SIGMA_TEMP_MAX} °C."
            )
        if df.empty:
            st.error("В загруженном файле нет точек внутри допустимого диапазона.")
            return

    tab_all, tab_grain = st.tabs(["Общая модель", "По номеру зерна"])
    with tab_all:
        render_model_block(df, "Модель по всем данным")
    with tab_grain:
        G_sel = st.selectbox("Номер зерна", sorted(df["G"].unique()))
        df_g = df[df["G"] == G_sel].copy()
        if df_g.empty:
            st.info("Нет точек для выбранного номера зерна")
        else:
            render_model_block(df_g, f"Модель для зерна G = {G_sel}")


if __name__ == "__main__":
    main()
