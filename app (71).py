from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="Регрессия экспериментальных точек", layout="wide")
plt.style.use("ggplot")


TARGET_ALIASES = [
    "t",
    "temp",
    "temperature",
    "температура",
    "temperature_c",
    "t_c",
    "t c",
]
D_ALIASES = [
    "d",
    "diameter",
    "particle_diameter",
    "equivalent_diameter",
    "эквивалентный диаметр",
    "диаметр",
    "dэкв",
    "dэкв мкм",
    "d экв",
]
TAU_ALIASES = [
    "tau",
    "τ",
    "time",
    "время",
    "duration",
    "tau_h",
    "tau h",
]
GRAIN_ALIASES = [
    "g",
    "grain",
    "grain_number",
    "номер зерна",
    "зерно",
]
SIGMA_ALIASES = [
    "csigma",
    "c_sigma",
    "sigma",
    "sigma_phase",
    "сигма",
    "содержание сигма-фазы",
    "c sigma",
    "c_sigma_pct",
    "c sigma pct",
]
ID_ALIASES = ["id", "sample", "sample_id", "образец", "точка"]


@dataclass
class FitResult:
    data: pd.DataFrame
    metrics: dict[str, float]
    params: pd.DataFrame
    weak_points: pd.DataFrame
    model_summary: str
    outlier_recommendation: pd.DataFrame


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("%", "")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("  ", " ")
    )


def find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for alias in aliases:
        alias_norm = normalize_name(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]
    for norm_name, original in normalized.items():
        if any(normalize_name(alias) in norm_name for alias in aliases):
            return original
    return None


def load_file(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    raw = uploaded_file.getvalue()
    bio = BytesIO(raw)

    if suffix in {"xls", "xlsx"}:
        excel = pd.ExcelFile(bio)
        sheets: list[pd.DataFrame] = []
        for sheet_name in excel.sheet_names:
            sheet_df = pd.read_excel(BytesIO(raw), sheet_name=sheet_name)
            if not sheet_df.dropna(how="all").empty and len(sheet_df.columns) > 0:
                sheet_df["_sheet_name"] = sheet_name
                sheets.append(sheet_df)
        if not sheets:
            raise ValueError("В файле Excel не найдено листов с данными.")
        return pd.concat(sheets, ignore_index=True)

    if suffix == "csv":
        return pd.read_csv(bio)

    raise ValueError("Поддерживаются только файлы XLS, XLSX и CSV.")


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    mapping = {
        "T": find_column(df.columns, TARGET_ALIASES),
        "D": find_column(df.columns, D_ALIASES),
        "tau": find_column(df.columns, TAU_ALIASES),
        "G": find_column(df.columns, GRAIN_ALIASES),
        "c_sigma": find_column(df.columns, SIGMA_ALIASES),
        "point_id": find_column(df.columns, ID_ALIASES),
    }

    required_missing = [key for key in ["T", "D", "tau", "G", "c_sigma"] if mapping[key] is None]
    if required_missing:
        raise ValueError(
            "Не удалось автоматически распознать обязательные столбцы: "
            + ", ".join(required_missing)
            + ". Переименуйте столбцы в понятные названия, например T, D, tau, G, c_sigma."
        )

    prepared = pd.DataFrame(
        {
            "T": pd.to_numeric(df[mapping["T"]], errors="coerce"),
            "D": pd.to_numeric(df[mapping["D"]], errors="coerce"),
            "tau": pd.to_numeric(df[mapping["tau"]], errors="coerce"),
            "G": pd.to_numeric(df[mapping["G"]], errors="coerce"),
            "c_sigma": pd.to_numeric(df[mapping["c_sigma"]], errors="coerce"),
        }
    )

    if mapping["point_id"] is not None:
        prepared["point_id"] = df[mapping["point_id"]].astype(str)
    else:
        prepared["point_id"] = [f"Точка {i + 1}" for i in range(len(prepared))]

    extra_columns = [col for col in df.columns if col not in set(mapping.values()) - {None}]
    for col in extra_columns:
        prepared[col] = df[col]

    prepared = prepared.dropna(subset=["T", "D", "tau", "G", "c_sigma"]).copy()
    prepared = prepared[(prepared["D"] > 0) & (prepared["tau"] > 0) & (prepared["c_sigma"] > 0)].copy()
    prepared = prepared[prepared["T"] > -273.15].copy()

    if prepared.empty:
        raise ValueError("После очистки не осталось корректных строк. Проверьте данные и единицы измерения.")

    prepared["T_kelvin"] = prepared["T"] + 273.15
    prepared["inv_T"] = 1.0 / prepared["T_kelvin"]
    prepared["ln_D"] = np.log(prepared["D"])
    prepared["ln_tau"] = np.log(prepared["tau"])
    prepared["ln_c_sigma"] = np.log(prepared["c_sigma"])

    return prepared.reset_index(drop=True)


def approximation_reliability(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    numerator = np.sum(np.square(y_true - y_pred))
    if denominator == 0:
        return np.nan
    return (1 - numerator / denominator) * 100


def build_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["T"]
    y_pred = df["T_pred"]
    abs_err = np.abs(df["abs_error"])
    rel_err = np.abs(df["rel_error_pct"])

    n = len(df)
    p = 5
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (max(n - p - 1, 1))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(rel_err.mean())
    max_err = float(abs_err.max())
    mean_err = float(df["error_celsius"].mean())
    std_err = float(df["error_celsius"].std(ddof=1)) if n > 1 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else np.nan
    ser = float(np.sqrt(np.sum(np.square(df["error_celsius"])) / max(n - p - 1, 1)))
    approx = float(approximation_reliability(y_true.to_numpy(), y_pred.to_numpy()))

    return {
        "Количество точек": float(n),
        "R²": float(r2),
        "Скорректированный R²": float(adj_r2),
        "RMSE, °C": rmse,
        "MAE, °C": mae,
        "MAPE, %": mape,
        "Среднее отклонение, °C": mean_err,
        "Стандартное отклонение ошибки, °C": std_err,
        "Максимальное отклонение, °C": max_err,
        "Стандартная ошибка регрессии": ser,
        "Корреляция факт/модель": corr,
        "Коэффициент достоверности аппроксимации, %": approx,
    }


def fit_model(df: pd.DataFrame) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    X = df[["ln_D", "ln_tau", "G", "ln_c_sigma"]]
    X = sm.add_constant(X)
    y = df["inv_T"]

    model = sm.OLS(y, X).fit()
    influence = model.get_influence()
    fitted_inv_t = model.predict(X)
    fitted_kelvin = 1.0 / fitted_inv_t
    fitted_c = fitted_kelvin - 273.15

    result_df = df.copy()
    result_df["inv_T_pred"] = fitted_inv_t
    result_df["T_pred"] = fitted_c
    result_df["error_celsius"] = result_df["T"] - result_df["T_pred"]
    result_df["abs_error"] = np.abs(result_df["error_celsius"])
    result_df["rel_error_pct"] = np.where(
        result_df["T"] != 0,
        result_df["abs_error"] / np.abs(result_df["T"]) * 100,
        np.nan,
    )
    result_df["standard_residual"] = influence.resid_studentized_internal
    result_df["leverage"] = influence.hat_matrix_diag
    result_df["cooks_distance"] = influence.cooks_distance[0]

    weak_points = result_df.sort_values(
        by=["abs_error", "cooks_distance", "standard_residual"], ascending=[False, False, False]
    ).copy()

    outlier_recommendation = weak_points[
        (weak_points["abs_error"] >= weak_points["abs_error"].quantile(0.9))
        | (np.abs(weak_points["standard_residual"]) > 2)
        | (weak_points["cooks_distance"] > 4 / len(result_df))
    ].copy()

    params = pd.DataFrame(
        {
            "Коэффициент": ["a", "b", "c", "d", "e"],
            "Параметр модели": ["const", "ln_D", "ln_tau", "G", "ln_c_sigma"],
            "Значение": [
                model.params.get("const", np.nan),
                model.params.get("ln_D", np.nan),
                model.params.get("ln_tau", np.nan),
                model.params.get("G", np.nan),
                model.params.get("ln_c_sigma", np.nan),
            ],
            "StdErr": [
                model.bse.get("const", np.nan),
                model.bse.get("ln_D", np.nan),
                model.bse.get("ln_tau", np.nan),
                model.bse.get("G", np.nan),
                model.bse.get("ln_c_sigma", np.nan),
            ],
            "t-статистика": [
                model.tvalues.get("const", np.nan),
                model.tvalues.get("ln_D", np.nan),
                model.tvalues.get("ln_tau", np.nan),
                model.tvalues.get("G", np.nan),
                model.tvalues.get("ln_c_sigma", np.nan),
            ],
            "p-value": [
                model.pvalues.get("const", np.nan),
                model.pvalues.get("ln_D", np.nan),
                model.pvalues.get("ln_tau", np.nan),
                model.pvalues.get("G", np.nan),
                model.pvalues.get("ln_c_sigma", np.nan),
            ],
            "Нижняя 95% граница": [
                model.conf_int().loc["const", 0],
                model.conf_int().loc["ln_D", 0],
                model.conf_int().loc["ln_tau", 0],
                model.conf_int().loc["G", 0],
                model.conf_int().loc["ln_c_sigma", 0],
            ],
            "Верхняя 95% граница": [
                model.conf_int().loc["const", 1],
                model.conf_int().loc["ln_D", 1],
                model.conf_int().loc["ln_tau", 1],
                model.conf_int().loc["G", 1],
                model.conf_int().loc["ln_c_sigma", 1],
            ],
        }
    )

    metrics = build_metrics(result_df)

    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=model.summary().as_text(),
        outlier_recommendation=outlier_recommendation,
    )


def metric_cards(metrics: dict[str, float]) -> None:
    keys = list(metrics.keys())
    cols = st.columns(4)
    for idx, key in enumerate(keys):
        value = metrics[key]
        with cols[idx % 4]:
            if np.isnan(value):
                st.metric(key, "—")
            elif abs(value) >= 100 or key == "Количество точек":
                st.metric(key, f"{value:,.0f}".replace(",", " "))
            else:
                st.metric(key, f"{value:,.4f}".replace(",", " "))


def scatter_fact_vs_pred(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["T"], df["T_pred"], color="#1f77b4", s=70, alpha=0.8)
    low = min(df["T"].min(), df["T_pred"].min())
    high = max(df["T"].max(), df["T_pred"].max())
    ax.plot([low, high], [low, high], "r--", linewidth=1.5, label="Идеальное совпадение")
    ax.set_xlabel("Экспериментальная температура, °C")
    ax.set_ylabel("Расчетная температура, °C")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def residual_plot(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.scatter(df["T_pred"], df["error_celsius"], color="#ff7f0e", s=70, alpha=0.8)
    ax.set_xlabel("Расчетная температура, °C")
    ax.set_ylabel("Ошибка (эксперимент - модель), °C")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def histogram_errors(df: pd.DataFrame, title: str) -> None:
    values = df["error_celsius"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = min(12, max(5, int(np.sqrt(len(values))))) if len(values) else 5
    ax.hist(values, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="black")
    ax.set_xlabel("Ошибка, °C")
    ax.set_ylabel("Частота")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def sigma_plot(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_df = df.sort_values("T")
    point_index = np.arange(1, len(sorted_df) + 1)
    ax.plot(point_index, sorted_df["T_pred"], color="#9467bd", linewidth=2, marker="o", label="Модель")
    ax.scatter(point_index, sorted_df["T"], color="#1f77b4", s=55, label="Эксперимент")
    ax.set_xlabel("Номер точки в порядке возрастания температуры")
    ax.set_ylabel("Температура, °C")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def qq_plot(df: pd.DataFrame, title: str) -> None:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    stats.probplot(df["error_celsius"], dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_result_block(result: FitResult, key_prefix: str = "main") -> None:
    st.subheader("Показатели качества модели")
    metric_cards(result.metrics)

    st.subheader("Коэффициенты модели")
    st.dataframe(result.params, use_container_width=True, hide_index=True)

    formula_parts = result.params.set_index("Коэффициент")["Значение"].to_dict()
    st.code(
        "1 / T(K) = "
        f"{formula_parts['a']:.8f} "
        f"+ ({formula_parts['b']:.8f})·ln(D) "
        f"+ ({formula_parts['c']:.8f})·ln(τ) "
        f"+ ({formula_parts['d']:.8f})·G "
        f"+ ({formula_parts['e']:.8f})·ln(cσ)",
        language="text",
    )

    st.subheader("Точки с наибольшим влиянием / отклонением")
    weak_view = result.weak_points[
        [
            "point_id",
            "T",
            "T_pred",
            "error_celsius",
            "rel_error_pct",
            "standard_residual",
            "cooks_distance",
            "G",
        ]
    ].head(15)
    st.dataframe(weak_view, use_container_width=True, hide_index=True)

    if not result.outlier_recommendation.empty:
        st.warning("Ниже точки, которые система рекомендует проверить или временно исключить из подгонки.")
        outlier_labels = result.outlier_recommendation["point_id"].astype(str).tolist()
        selected = st.multiselect(
            "Исключить точки из расчета",
            options=result.data["point_id"].astype(str).tolist(),
            default=outlier_labels,
            key=f"exclude_{key_prefix}",
        )
        if selected:
            filtered = result.data[~result.data["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                st.info(f"Пересчет после исключения {len(selected)} точек.")
                recalculated = fit_model(filtered)
                metric_cards(recalculated.metrics)
                st.dataframe(
                    recalculated.params,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.error("После исключения осталось слишком мало точек для устойчивой подгонки.")

    st.subheader("Таблица по всем точкам")
    view_columns = [
        "point_id",
        "T",
        "T_pred",
        "error_celsius",
        "abs_error",
        "rel_error_pct",
        "standard_residual",
        "cooks_distance",
        "D",
        "tau",
        "G",
        "c_sigma",
    ]
    st.dataframe(result.data[view_columns], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        scatter_fact_vs_pred(result.data, "Эксперимент vs расчет")
        residual_plot(result.data, "Остатки модели")
        qq_plot(result.data, "Q-Q график остатков")
    with c2:
        histogram_errors(result.data, "Распределение ошибок")
        sigma_plot(result.data, "Модель и экспериментальные точки")

    with st.expander("Подробная статистическая сводка (statsmodels)"):
        st.text(result.model_summary)


st.title("Подбор регрессионной модели для экспериментальных точек")
st.write(
    "Модель: 1 / T = a + b·ln(D) + c·ln(τ) + d·G + e·ln(cσ), где T внутри расчета переводится в Кельвины."
)

uploaded_file = st.file_uploader(
    "Загрузите файл с исходными данными",
    type=["xls", "xlsx", "csv"],
    help="Поддерживаются XLS/XLSX/CSV. Обязательные поля: T, D, tau, G, c_sigma.",
)

if uploaded_file is None:
    st.info("Загрузите файл с данными, и приложение сразу покажет модель, ошибки, слабые точки и графики.")
    st.stop()

try:
    raw_df = load_file(uploaded_file)
    prepared_df = prepare_dataframe(raw_df)
except Exception as exc:
    st.error(f"Не удалось обработать файл: {exc}")
    st.stop()

with st.expander("Предпросмотр исходных данных"):
    st.dataframe(raw_df, use_container_width=True)

st.success(f"Загружено корректных точек: {len(prepared_df)}")

main_tab, grain_tab = st.tabs(["Общая модель", "Модели по номерам зерна"])

with main_tab:
    try:
        result = fit_model(prepared_df)
        show_result_block(result, key_prefix="all")
    except Exception as exc:
        st.error(f"Не удалось построить общую модель: {exc}")

with grain_tab:
    grain_scores: list[dict[str, float]] = []
    grains = sorted(prepared_df["G"].dropna().unique().tolist())
    valid_grains = []
    for grain in grains:
        grain_df = prepared_df[prepared_df["G"] == grain].copy()
        if len(grain_df) >= 7:
            valid_grains.append(grain)

    if not valid_grains:
        st.warning("Для отдельных номеров зерна пока недостаточно точек. Нужно минимум 7 точек на номер зерна.")
    else:
        selected_grain = st.selectbox("Выберите номер зерна", valid_grains)
        for grain in valid_grains:
            grain_df = prepared_df[prepared_df["G"] == grain].copy()
            try:
                grain_result = fit_model(grain_df)
                grain_scores.append(
                    {
                        "Номер зерна": grain,
                        "Количество точек": grain_result.metrics["Количество точек"],
                        "R²": grain_result.metrics["R²"],
                        "RMSE, °C": grain_result.metrics["RMSE, °C"],
                        "MAE, °C": grain_result.metrics["MAE, °C"],
                        "MAPE, %": grain_result.metrics["MAPE, %"],
                        "Коэффициент достоверности аппроксимации, %": grain_result.metrics[
                            "Коэффициент достоверности аппроксимации, %"
                        ],
                    }
                )
                if grain == selected_grain:
                    show_result_block(grain_result, key_prefix=f"grain_{grain}")
            except Exception:
                continue

        if grain_scores:
            st.subheader("Сравнение качества модели по номерам зерна")
            score_df = pd.DataFrame(grain_scores).sort_values(
                by=["R²", "RMSE, °C", "MAPE, %"],
                ascending=[False, True, True],
            )
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            best_grain = score_df.iloc[0]
            st.info(
                f"Лучше всего модель выглядит для номера зерна {best_grain['Номер зерна']}: "
                f"R²={best_grain['R²']:.4f}, RMSE={best_grain['RMSE, °C']:.4f} °C."
            )
