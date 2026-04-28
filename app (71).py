from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy import optimize, stats
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

SIGMA_SATURATION_LIMIT = 18.0
REAL_WORLD_POINT = {
    "tau": 150000.0,
    "D": 7.9,
    "c_sigma": 10.18,
    "G": 10.0,
    "temp_min": 570.0,
    "temp_max": 600.0,
}

GRAIN_SIZE_MM = {
    3.0: 0.125,
    4.0: 0.088,
    5.0: 0.062,
    6.0: 0.044,
    7.0: 0.031,
    8.0: 0.022,
    9.0: 0.015,
    10.0: 0.011,
}

SIGMA_UNIVERSAL_GRAINS = [8.0, 9.0, 10.0]

SCIENTIFIC_UNIVERSAL_SIGMA_PARAGRAPH = (
    "Универсализированная модель содержания σ-фазы по размеру зерна строится по наиболее надежной части "
    "экспериментальной выборки, где содержание σ-фазы достаточно велико для устойчивого измерения и, "
    "следовательно, коэффициенты локальных зависимостей определяются с меньшим влиянием случайного шума. "
    "В этой постановке отдельные зерновые модели сначала подбираются независимо, после чего их коэффициенты "
    "рассматриваются как функции физического размера зерна. Такой подход позволяет получить не формально "
    "универсальную зависимость для всех возможных зерен, а физически и статистически обоснованную "
    "интерполяционную метамодель в области качественных данных, пригодную для осторожной экстраполяции на "
    "соседние зеренные состояния."
)


@dataclass
class FitResult:
    data: pd.DataFrame
    metrics: dict[str, float]
    params: pd.DataFrame
    weak_points: pd.DataFrame
    model_summary: str
    outlier_recommendation: pd.DataFrame
    formula_text: str
    model_label: str


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
    sigma_clipped = np.clip(prepared["c_sigma"], 1e-9, SIGMA_SATURATION_LIMIT - 1e-9)
    prepared["sigma_remaining"] = SIGMA_SATURATION_LIMIT - sigma_clipped
    prepared["sigma_remaining_fraction"] = prepared["sigma_remaining"] / SIGMA_SATURATION_LIMIT
    prepared["ln_sigma_remaining_fraction"] = np.log(prepared["sigma_remaining_fraction"])
    prepared["sigma_saturation_logit"] = np.log(sigma_clipped / prepared["sigma_remaining"])

    return prepared.reset_index(drop=True)


def sigma_saturation_feature(c_sigma: float, sigma_limit: float = SIGMA_SATURATION_LIMIT) -> float:
    if c_sigma <= 0:
        raise ValueError("Содержание сигма-фазы должно быть больше нуля.")
    if c_sigma >= sigma_limit:
        raise ValueError(
            f"Содержание сигма-фазы должно быть меньше предельного уровня {sigma_limit:.2f}% для насыщаемой модели."
        )
    return float(np.log(c_sigma / (sigma_limit - c_sigma)))


def sigma_remaining_feature(c_sigma: float, sigma_limit: float = SIGMA_SATURATION_LIMIT) -> float:
    if c_sigma <= 0:
        raise ValueError("Содержание сигма-фазы должно быть больше нуля.")
    if c_sigma >= sigma_limit:
        raise ValueError(
            f"Содержание сигма-фазы должно быть меньше предельного уровня {sigma_limit:.2f}% для кинетической модели."
        )
    return float(np.log((sigma_limit - c_sigma) / sigma_limit))


def approximation_reliability(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    numerator = np.sum(np.square(y_true - y_pred))
    if denominator == 0:
        return np.nan
    return (1 - numerator / denominator) * 100


def build_metrics(df: pd.DataFrame, predictor_count: int) -> dict[str, float]:
    y_true = df["T"]
    y_pred = df["T_pred"]
    abs_err = np.abs(df["abs_error"])
    rel_err = np.abs(df["rel_error_pct"])

    n = len(df)
    p = predictor_count
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


def fit_engineering_model(df: pd.DataFrame, include_grain: bool = True) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    feature_columns = ["ln_D", "ln_tau", "ln_c_sigma"]
    if include_grain:
        feature_columns.insert(2, "G")

    X = df[feature_columns]
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

    coeff_labels = [
        ("a", "const"),
        ("b", "ln_D"),
        ("c", "ln_tau"),
    ]
    if include_grain:
        coeff_labels.append(("d", "G"))
        coeff_labels.append(("e", "ln_c_sigma"))
    else:
        coeff_labels.append(("d", "ln_c_sigma"))

    conf_int = model.conf_int()
    params = pd.DataFrame(
        {
            "Коэффициент": [label for label, _ in coeff_labels],
            "Параметр модели": [param for _, param in coeff_labels],
            "Значение": [model.params.get(param, np.nan) for _, param in coeff_labels],
            "StdErr": [model.bse.get(param, np.nan) for _, param in coeff_labels],
            "t-статистика": [model.tvalues.get(param, np.nan) for _, param in coeff_labels],
            "p-value": [model.pvalues.get(param, np.nan) for _, param in coeff_labels],
            "Нижняя 95% граница": [conf_int.loc[param, 0] for _, param in coeff_labels],
            "Верхняя 95% граница": [conf_int.loc[param, 1] for _, param in coeff_labels],
        }
    )

    metrics = build_metrics(result_df, predictor_count=len(feature_columns))

    formula_text = (
        "1 / T(K) = "
        f"{model.params.get('const', np.nan):.8f} "
        f"+ ({model.params.get('ln_D', np.nan):.8f})·ln(D) "
        f"+ ({model.params.get('ln_tau', np.nan):.8f})·ln(τ) "
    )
    if include_grain:
        formula_text += f"+ ({model.params.get('G', np.nan):.8f})·G "
    formula_text += f"+ ({model.params.get('ln_c_sigma', np.nan):.8f})·ln(cσ)"

    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=model.summary().as_text(),
        outlier_recommendation=outlier_recommendation,
        formula_text=formula_text,
        model_label="Базовая инженерная модель",
    )


def fit_improved_model(df: pd.DataFrame, include_grain: bool = True) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    feature_columns = ["ln_tau", "inv_T", "ln_c_sigma"]
    if include_grain:
        feature_columns.insert(2, "G")

    X = sm.add_constant(df[feature_columns])
    y = df["ln_D"]

    model = sm.OLS(y, X).fit()
    influence = model.get_influence()

    a2 = model.params.get("inv_T", np.nan)
    if not np.isfinite(a2) or abs(a2) < 1e-12:
        raise ValueError(
            "Коэффициент при 1/T в улучшенной модели оказался слишком мал. Невозможно устойчиво восстановить температуру."
        )

    numerator = (
        df["ln_D"]
        - model.params.get("const", 0.0)
        - model.params.get("ln_tau", 0.0) * df["ln_tau"]
        - model.params.get("ln_c_sigma", 0.0) * df["ln_c_sigma"]
    )
    if include_grain:
        numerator = numerator - model.params.get("G", 0.0) * df["G"]

    fitted_inv_t = numerator / a2
    if np.any(fitted_inv_t <= 0):
        raise ValueError(
            "Улучшенная модель дала неположительные значения 1/T. Проверьте диапазон данных или исключите выбросы."
        )

    fitted_kelvin = 1.0 / fitted_inv_t
    fitted_c = fitted_kelvin - 273.15

    result_df = df.copy()
    result_df["ln_D_pred"] = model.predict(X)
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

    coeff_labels = [
        ("a0", "const"),
        ("a1", "ln_tau"),
        ("a2", "inv_T"),
    ]
    if include_grain:
        coeff_labels.append(("a3", "G"))
        coeff_labels.append(("a4", "ln_c_sigma"))
    else:
        coeff_labels.append(("a3", "ln_c_sigma"))

    conf_int = model.conf_int()
    params = pd.DataFrame(
        {
            "Коэффициент": [label for label, _ in coeff_labels],
            "Параметр модели": [param for _, param in coeff_labels],
            "Значение": [model.params.get(param, np.nan) for _, param in coeff_labels],
            "StdErr": [model.bse.get(param, np.nan) for _, param in coeff_labels],
            "t-статистика": [model.tvalues.get(param, np.nan) for _, param in coeff_labels],
            "p-value": [model.pvalues.get(param, np.nan) for _, param in coeff_labels],
            "Нижняя 95% граница": [conf_int.loc[param, 0] for _, param in coeff_labels],
            "Верхняя 95% граница": [conf_int.loc[param, 1] for _, param in coeff_labels],
        }
    )

    metrics = build_metrics(result_df, predictor_count=len(feature_columns))

    formula_text = (
        "ln(D) = "
        f"{model.params.get('const', np.nan):.8f} "
        f"+ ({model.params.get('ln_tau', np.nan):.8f})·ln(τ) "
        f"+ ({model.params.get('inv_T', np.nan):.8f})·(1/T(K)) "
    )
    if include_grain:
        formula_text += f"+ ({model.params.get('G', np.nan):.8f})·G "
    formula_text += f"+ ({model.params.get('ln_c_sigma', np.nan):.8f})·ln(cσ)"

    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=model.summary().as_text(),
        outlier_recommendation=outlier_recommendation,
        formula_text=formula_text,
        model_label="Улучшенная физически ориентированная модель",
    )


def fit_diameter_growth_model(df: pd.DataFrame, include_grain: bool = False) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    if include_grain:
        result_frames: list[pd.DataFrame] = []
        param_rows: list[dict[str, float | str]] = []
        summary_parts: list[str] = []

        for grain_value in sorted(df["G"].dropna().unique().tolist()):
            grain_df = df[df["G"] == grain_value].copy()
            if len(grain_df) < 7:
                summary_parts.append(f"Зерно {grain_value}: пропущено, точек меньше 7.")
                continue
            grain_result = fit_diameter_growth_model(grain_df, include_grain=False)
            result_frames.append(grain_result.data)
            summary_parts.append(f"--- Модель роста диаметра для зерна {grain_value} ---\n{grain_result.model_summary}")
            for _, row in grain_result.params.iterrows():
                param_rows.append(
                    {
                        "Коэффициент": f"{row['Коэффициент']}(G={grain_value})",
                        "Параметр модели": f"grain_{grain_value}_{row['Параметр модели']}",
                        "Значение": row["Значение"],
                        "StdErr": row["StdErr"],
                        "t-статистика": row["t-статистика"],
                        "p-value": row["p-value"],
                        "Нижняя 95% граница": row["Нижняя 95% граница"],
                        "Верхняя 95% граница": row["Верхняя 95% граница"],
                    }
                )

        if not result_frames:
            raise ValueError("Не удалось построить ни одной модели роста диаметра по отдельным зернам: недостаточно данных.")

        result_df = pd.concat(result_frames, ignore_index=True)
        params = pd.DataFrame(param_rows)
        metrics = build_metrics(result_df, predictor_count=2)

        real_grain = REAL_WORLD_POINT["G"]
        matching = params[params["Параметр модели"].str.startswith(f"grain_{real_grain}_")]
        if matching.empty:
            metrics["Прогноз для реальной точки, °C"] = np.nan
            metrics["Отклонение реальной точки от диапазона, °C"] = np.nan
        else:
            grain_params = {
                row["Параметр модели"].split(f"grain_{real_grain}_", 1)[1]: row["Значение"]
                for _, row in matching.iterrows()
            }
            real_temp = predict_temperature_diameter_growth(grain_params, REAL_WORLD_POINT["D"], REAL_WORLD_POINT["tau"])
            metrics["Прогноз для реальной точки, °C"] = float(real_temp)
            if REAL_WORLD_POINT["temp_min"] <= real_temp <= REAL_WORLD_POINT["temp_max"]:
                metrics["Отклонение реальной точки от диапазона, °C"] = 0.0
            elif real_temp < REAL_WORLD_POINT["temp_min"]:
                metrics["Отклонение реальной точки от диапазона, °C"] = REAL_WORLD_POINT["temp_min"] - real_temp
            else:
                metrics["Отклонение реальной точки от диапазона, °C"] = real_temp - REAL_WORLD_POINT["temp_max"]

        weak_points = result_df.sort_values(
            by=["abs_error", "cooks_distance", "standard_residual"], ascending=[False, False, False]
        ).copy()
        outlier_recommendation = weak_points[
            (weak_points["abs_error"] >= weak_points["abs_error"].quantile(0.9))
            | (np.abs(weak_points["standard_residual"]) > 2)
        ].copy()
        return FitResult(
            data=result_df,
            metrics=metrics,
            params=params,
            weak_points=weak_points,
            model_summary="Модели роста диаметра по отдельным зернам.\n\n" + "\n\n".join(summary_parts),
            outlier_recommendation=outlier_recommendation,
            formula_text="Для каждого номера зерна отдельно: ln(D)=a_G+b_G·ln(τ)+c_G·(1/T(K))",
            model_label="Эмпирические модели роста диаметра по отдельным зернам",
        )

    feature_columns = ["ln_tau", "inv_T"]
    X = sm.add_constant(df[feature_columns])
    y = df["ln_D"]

    model = sm.OLS(y, X).fit()
    influence = model.get_influence()

    a2 = model.params.get("inv_T", np.nan)
    if not np.isfinite(a2) or abs(a2) < 1e-12:
        raise ValueError("Коэффициент при 1/T в модели роста диаметра слишком мал для устойчивого обратного расчета.")

    numerator = df["ln_D"] - model.params.get("const", 0.0) - model.params.get("ln_tau", 0.0) * df["ln_tau"]
    inv_t_pred = numerator / a2
    if np.any(inv_t_pred <= 0):
        raise ValueError("Модель роста диаметра дала неположительное значение 1/T для части точек.")

    temp_kelvin_pred = 1.0 / inv_t_pred
    temp_c_pred = temp_kelvin_pred - 273.15

    result_df = df.copy()
    result_df["inv_T_pred"] = inv_t_pred
    result_df["T_pred"] = temp_c_pred
    result_df["error_celsius"] = result_df["T"] - result_df["T_pred"]
    result_df["abs_error"] = np.abs(result_df["error_celsius"])
    result_df["rel_error_pct"] = np.where(result_df["T"] != 0, result_df["abs_error"] / np.abs(result_df["T"]) * 100, np.nan)
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

    conf_int = model.conf_int()
    params = pd.DataFrame(
        {
            "Коэффициент": ["a", "b", "c"],
            "Параметр модели": ["const", "ln_tau", "inv_T"],
            "Значение": [model.params.get("const", np.nan), model.params.get("ln_tau", np.nan), model.params.get("inv_T", np.nan)],
            "StdErr": [model.bse.get("const", np.nan), model.bse.get("ln_tau", np.nan), model.bse.get("inv_T", np.nan)],
            "t-статистика": [model.tvalues.get("const", np.nan), model.tvalues.get("ln_tau", np.nan), model.tvalues.get("inv_T", np.nan)],
            "p-value": [model.pvalues.get("const", np.nan), model.pvalues.get("ln_tau", np.nan), model.pvalues.get("inv_T", np.nan)],
            "Нижняя 95% граница": [conf_int.loc["const", 0], conf_int.loc["ln_tau", 0], conf_int.loc["inv_T", 0]],
            "Верхняя 95% граница": [conf_int.loc["const", 1], conf_int.loc["ln_tau", 1], conf_int.loc["inv_T", 1]],
        }
    )

    metrics = build_metrics(result_df, predictor_count=len(feature_columns))
    formula_text = (
        "ln(D) = a + b·ln(τ) + c·(1/T(K))\n"
        f"a = {model.params.get('const', np.nan):.8f}\n"
        f"b = {model.params.get('ln_tau', np.nan):.8f}\n"
        f"c = {model.params.get('inv_T', np.nan):.8f}\n"
        f"Итог: ln(D) = {model.params.get('const', np.nan):.8f} + ({model.params.get('ln_tau', np.nan):.8f})·ln(τ) + ({model.params.get('inv_T', np.nan):.8f})·(1/T(K))"
    )

    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=model.summary().as_text(),
        outlier_recommendation=outlier_recommendation,
        formula_text=formula_text,
        model_label="Эмпирическая модель роста диаметра ln(D)=a+b·ln(τ)+c·(1/T)",
    )


def fit_anchor_saturation_model(df: pd.DataFrame, include_grain: bool = True) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    def sigma_power_model(params: np.ndarray, tau_vals: np.ndarray, temp_vals: np.ndarray) -> np.ndarray:
        log_a, p_exp, m_exp = params
        tau_term = np.power(np.maximum(tau_vals, 1e-12), p_exp)
        temp_term = np.power(np.clip((temp_vals - 550.0) / 350.0, 1e-9, None), m_exp)
        return np.exp(log_a) * tau_term * temp_term

    def fit_single_grain(grain_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_vals = grain_df["tau"].to_numpy(dtype=float)
        temp_vals = grain_df["T"].to_numpy(dtype=float)
        sigma_true = grain_df["c_sigma"].to_numpy(dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            pred = sigma_power_model(params, tau_vals, temp_vals)
            penalty = 1e-5 * np.sum(np.square(params))
            return np.append(pred - sigma_true, np.sqrt(penalty))

        fit = optimize.least_squares(
            residuals,
            x0=np.array([-8.0, 0.35, 0.7], dtype=float),
            bounds=(np.array([-30.0, 0.01, 0.01], dtype=float), np.array([10.0, 1.5, 4.0], dtype=float)),
            method="trf",
            loss="soft_l1",
            max_nfev=20000,
        )
        if not fit.success:
            raise ValueError(f"Подгонка степенной sigma-модели не сошлась: {fit.message}")

        params_vec = fit.x
        sigma_pred = np.clip(sigma_power_model(params_vec, tau_vals, temp_vals), 0.0, SIGMA_SATURATION_LIMIT)
        log_a, p_exp, m_exp = params_vec
        denom = np.exp(log_a) * np.power(np.maximum(tau_vals, 1e-12), p_exp)
        temp_norm_pred = np.power(np.maximum(sigma_true / np.maximum(denom, 1e-12), 1e-12), 1.0 / m_exp)
        temp_pred = np.clip(550.0 + 350.0 * temp_norm_pred, 550.0, 900.0)
        return params_vec, sigma_pred, temp_pred

    def solve_temp_from_model(params: dict[str, float], tau_value: float, sigma_value: float) -> float:
        if sigma_value <= 0:
            raise ValueError("Содержание сигма-фазы должно быть больше нуля.")
        log_a = params["log_a"]
        p_exp = params["p_exp"]
        m_exp = params["m_exp"]
        denom = np.exp(log_a) * np.power(max(tau_value, 1e-12), p_exp)
        if not np.isfinite(denom) or denom <= 0 or not np.isfinite(m_exp) or abs(m_exp) < 1e-12:
            raise ValueError("Степенная sigma-модель дала некорректные параметры.")
        temp_norm = np.power(max(sigma_value / denom, 1e-12), 1.0 / m_exp)
        return float(np.clip(550.0 + 350.0 * temp_norm, 550.0, 900.0))

    if include_grain:
        result_frames: list[pd.DataFrame] = []
        param_rows: list[dict[str, float | str]] = []
        summary_parts: list[str] = []

        for grain_value in sorted(df["G"].dropna().unique().tolist()):
            grain_df = df[df["G"] == grain_value].copy()
            if len(grain_df) < 7:
                summary_parts.append(f"Зерно {grain_value}: пропущено, точек меньше 7.")
                continue
            grain_result = fit_anchor_saturation_model(grain_df, include_grain=False)
            result_frames.append(grain_result.data)
            summary_parts.append(f"--- Модель для зерна {grain_value} ---\n{grain_result.model_summary}")
            for _, row in grain_result.params.iterrows():
                param_rows.append(
                    {
                        "Коэффициент": f"{row['Коэффициент']}(G={grain_value})",
                        "Параметр модели": f"grain_{grain_value}_{row['Параметр модели']}",
                        "Значение": row["Значение"],
                        "StdErr": row["StdErr"],
                        "t-статистика": row["t-статистика"],
                        "p-value": row["p-value"],
                        "Нижняя 95% граница": row["Нижняя 95% граница"],
                        "Верхняя 95% граница": row["Верхняя 95% граница"],
                    }
                )

        if not result_frames:
            raise ValueError("Не удалось построить ни одной модели по отдельным зернам: недостаточно данных.")

        result_df = pd.concat(result_frames, ignore_index=True)

        params = pd.DataFrame(param_rows)
        metrics = build_metrics(result_df, predictor_count=3)
        metrics["RMSE модели сигма-фазы, %"] = float(np.sqrt(mean_squared_error(result_df["c_sigma"], result_df["sigma_pred_pct"])))

        real_grain = REAL_WORLD_POINT["G"]
        matching = params[params["Параметр модели"].str.startswith(f"grain_{real_grain}_")]
        if matching.empty:
            metrics["Прогноз для реальной точки, °C"] = np.nan
            metrics["Отклонение реальной точки от диапазона, °C"] = np.nan
        else:
            grain_params = {
                row["Параметр модели"].split(f"grain_{real_grain}_", 1)[1]: row["Значение"]
                for _, row in matching.iterrows()
            }
            real_temp = solve_temp_from_model(grain_params, REAL_WORLD_POINT["tau"], REAL_WORLD_POINT["c_sigma"])
            metrics["Прогноз для реальной точки, °C"] = float(real_temp)
            if REAL_WORLD_POINT["temp_min"] <= real_temp <= REAL_WORLD_POINT["temp_max"]:
                metrics["Отклонение реальной точки от диапазона, °C"] = 0.0
            elif real_temp < REAL_WORLD_POINT["temp_min"]:
                metrics["Отклонение реальной точки от диапазона, °C"] = REAL_WORLD_POINT["temp_min"] - real_temp
            else:
                metrics["Отклонение реальной точки от диапазона, °C"] = real_temp - REAL_WORLD_POINT["temp_max"]

        weak_points = result_df.sort_values(by=["abs_error", "standard_residual"], ascending=[False, False]).copy()
        outlier_recommendation = weak_points[
            (weak_points["abs_error"] >= weak_points["abs_error"].quantile(0.9))
            | (np.abs(weak_points["standard_residual"]) > 2)
        ].copy()

        formula_text = (
            "Для каждого номера зерна отдельно:\n"
            "cσ = A_G · τ^p_G · ((T - 550) / 350)^m_G\n"
            "Температура для проверки восстанавливается обратным степенным пересчетом."
        )
        summary_text = (
            "Прямая степенная sigma-модель по каждому номеру зерна отдельно.\n\n"
            + "\n\n".join(summary_parts)
        )
        return FitResult(
            data=result_df,
            metrics=metrics,
            params=params,
            weak_points=weak_points,
            model_summary=summary_text,
            outlier_recommendation=outlier_recommendation,
            formula_text=formula_text,
            model_label="Прямая степенная sigma-модель по отдельным зернам",
        )

    params_vec, pred_sigma, temp_pred = fit_single_grain(df.copy())
    log_a, p_exp, m_exp = params_vec
    result_df = df.copy()
    result_df["inv_T_pred"] = 1.0 / (temp_pred + 273.15)
    result_df["T_pred"] = temp_pred
    result_df["sigma_pred_pct"] = pred_sigma
    result_df["error_celsius"] = result_df["T"] - result_df["T_pred"]
    result_df["abs_error"] = np.abs(result_df["error_celsius"])
    result_df["rel_error_pct"] = np.where(result_df["T"] != 0, result_df["abs_error"] / np.abs(result_df["T"]) * 100, np.nan)
    std_err = result_df["error_celsius"].std(ddof=1)
    result_df["standard_residual"] = (
        (result_df["error_celsius"] - result_df["error_celsius"].mean()) / std_err if np.isfinite(std_err) and std_err > 0 else 0.0
    )
    result_df["leverage"] = np.nan
    result_df["cooks_distance"] = np.nan
    weak_points = result_df.sort_values(by=["abs_error", "standard_residual"], ascending=[False, False]).copy()
    outlier_recommendation = weak_points[
        (weak_points["abs_error"] >= weak_points["abs_error"].quantile(0.9)) | (np.abs(weak_points["standard_residual"]) > 2)
    ].copy()
    params = pd.DataFrame(
        {
            "Коэффициент": ["A", "p", "m"],
            "Параметр модели": ["log_a", "p_exp", "m_exp"],
            "Значение": [log_a, p_exp, m_exp],
            "StdErr": [np.nan, np.nan, np.nan],
            "t-статистика": [np.nan, np.nan, np.nan],
            "p-value": [np.nan, np.nan, np.nan],
            "Нижняя 95% граница": [np.nan, np.nan, np.nan],
            "Верхняя 95% граница": [np.nan, np.nan, np.nan],
        }
    )
    metrics = build_metrics(result_df, predictor_count=3)
    metrics["RMSE модели сигма-фазы, %"] = float(np.sqrt(mean_squared_error(result_df["c_sigma"], result_df["sigma_pred_pct"])))
    weak_points = result_df.sort_values(by=["abs_error", "standard_residual"], ascending=[False, False]).copy()
    formula_text = (
        "cσ = A · τ^p · ((T - 550) / 350)^m\n"
        f"A = exp({log_a:.8f}) = {np.exp(log_a):.8f}\n"
        f"p = {p_exp:.8f}\n"
        f"m = {m_exp:.8f}\n"
        f"Итог: cσ = {np.exp(log_a):.8f} · τ^{p_exp:.8f} · ((T - 550) / 350)^{m_exp:.8f}"
    )
    summary_text = (
        "Прямая степенная sigma-модель для одного зерна.\n"
        f"Параметры: log(A)={log_a:.6f}, p={p_exp:.6f}, m={m_exp:.6f}."
    )
    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=summary_text,
        outlier_recommendation=outlier_recommendation,
        formula_text=formula_text,
        model_label="Прямая степенная sigma-модель для одного зерна",
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


def sigma_metric_summary(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["c_sigma"].to_numpy(dtype=float)
    y_pred = df["sigma_pred_pct"].to_numpy(dtype=float)
    sigma_error = y_true - y_pred
    return {
        "Количество точек": float(len(df)),
        "R² по cσ": float(r2_score(y_true, y_pred)) if len(df) >= 2 else np.nan,
        "RMSE по cσ, %": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE по cσ, %": float(mean_absolute_error(y_true, y_pred)),
        "MAPE по cσ, %": float(np.mean(np.abs(sigma_error) / np.maximum(np.abs(y_true), 1e-9)) * 100.0),
        "Корреляция факт/модель по cσ": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(df) >= 2 else np.nan,
    }


def temperature_metric_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "Количество точек": float(len(df)),
        "R² по T": float(r2_score(df["T"], df["T_pred"])) if len(df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(df["T"], df["T_pred"]))),
        "MAE по T, °C": float(mean_absolute_error(df["T"], df["T_pred"])),
        "MAPE по T, %": float(np.mean(np.abs(df["error_celsius"]) / np.maximum(np.abs(df["T"]), 1e-9)) * 100.0),
        "Корреляция факт/модель по T": float(np.corrcoef(df["T"], df["T_pred"])[0, 1]) if len(df) >= 2 else np.nan,
    }


def sigma_scatter_fact_vs_pred(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["c_sigma"], df["sigma_pred_pct"], color="#1f77b4", s=70, alpha=0.8)
    low = min(df["c_sigma"].min(), df["sigma_pred_pct"].min())
    high = max(df["c_sigma"].max(), df["sigma_pred_pct"].max())
    ax.plot([low, high], [low, high], "r--", linewidth=1.5, label="Идеальное совпадение")
    ax.set_xlabel("Экспериментальное cσ, %")
    ax.set_ylabel("Расчетное cσ, %")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def sigma_vs_temperature_plot(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_df = df.sort_values("T")
    ax.scatter(sorted_df["T"], sorted_df["c_sigma"], color="#1f77b4", s=60, label="Эксперимент")
    ax.plot(sorted_df["T"], sorted_df["sigma_pred_pct"], color="#d62728", linewidth=2, marker="o", label="Модель")
    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Содержание сигма-фазы cσ, %")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def sigma_vs_time_plot(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_df = df.sort_values("tau")
    ax.scatter(sorted_df["tau"], sorted_df["c_sigma"], color="#1f77b4", s=60, label="Эксперимент")
    ax.plot(sorted_df["tau"], sorted_df["sigma_pred_pct"], color="#2ca02c", linewidth=2, marker="o", label="Модель")
    ax.set_xscale("log")
    ax.set_xlabel("Время τ, ч (log)")
    ax.set_ylabel("Содержание сигма-фазы cσ, %")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_sigma_grain_block(result: FitResult, grain_value: float) -> None:
    st.subheader(f"Sigma-модель для номера зерна {grain_value}")
    st.caption("Сначала показана предсказательность модели по температуре, ниже — качество прямой подгонки по содержанию сигма-фазы.")
    st.subheader("Качество предсказания температуры")
    metric_cards(temperature_metric_summary(result.data))
    st.subheader("Качество подгонки по содержанию сигма-фазы")
    sigma_metrics = sigma_metric_summary(result.data)
    metric_cards(sigma_metrics)
    st.subheader("Коэффициенты модели")
    st.dataframe(result.params, use_container_width=True, hide_index=True)
    st.caption(result.model_label)
    st.code(result.formula_text, language="text")

    st.subheader("Калькулятор температуры для этого номера зерна")
    calc_params = result.params.set_index("Параметр модели")["Значение"].to_dict()
    with st.form(key=f"sigma_grain_form_{grain_value}"):
        c1, c2 = st.columns(2)
        with c1:
            tau_value = st.number_input(
                f"Время наработки τ для зерна {grain_value}",
                min_value=1.0,
                value=1000.0,
                step=1.0,
                format="%.0f",
                key=f"sigma_grain_tau_{grain_value}",
            )
        with c2:
            sigma_value = st.number_input(
                f"Содержание сигма-фазы cσ для зерна {grain_value}, %",
                min_value=0.01,
                value=1.0,
                step=0.01,
                format="%.2f",
                key=f"sigma_grain_sigma_{grain_value}",
            )
        submitted = st.form_submit_button("Рассчитать")
    if submitted:
        try:
            calc_temp = predict_temperature_anchor_saturation(calc_params, 1.0, tau_value, sigma_value, grain_value)
            st.metric("Расчетная температура, °C", f"{calc_temp:.4f}")
        except Exception as exc:
            st.error(f"Не удалось выполнить расчет температуры для этого зерна: {exc}")

    if not result.outlier_recommendation.empty:
        st.warning("Ниже точки, которые система рекомендует проверить или временно исключить из подгонки sigma-модели.")
        outlier_labels = result.outlier_recommendation["point_id"].astype(str).tolist()
        selected = st.multiselect(
            "Исключить точки из sigma-модели",
            options=result.data["point_id"].astype(str).tolist(),
            default=outlier_labels,
            key=f"exclude_sigma_grain_{grain_value}",
        )
        if selected:
            filtered = result.data[~result.data["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                st.info(f"Пересчет sigma-модели после исключения {len(selected)} точек.")
                recalculated = fit_anchor_saturation_model(filtered, include_grain=False)
                st.subheader("Качество предсказания температуры после пересчета")
                metric_cards(temperature_metric_summary(recalculated.data))
                st.subheader("Качество подгонки по содержанию сигма-фазы после пересчета")
                metric_cards(sigma_metric_summary(recalculated.data))
                st.dataframe(recalculated.params, use_container_width=True, hide_index=True)
                st.code(recalculated.formula_text, language="text")
                result = recalculated
            else:
                st.error("После исключения осталось слишком мало точек для устойчивой подгонки sigma-модели.")

    st.subheader("Таблица по точкам")
    sigma_view = result.data[["point_id", "T", "tau", "G", "c_sigma", "sigma_pred_pct", "T_pred", "error_celsius"]].copy()
    sigma_view["Ошибка по cσ, %"] = sigma_view["c_sigma"] - sigma_view["sigma_pred_pct"]
    st.dataframe(sigma_view, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        sigma_scatter_fact_vs_pred(result.data, "Эксперимент vs модель по cσ")
        sigma_vs_temperature_plot(result.data, "Зависимость cσ от температуры")
    with c2:
        sigma_vs_time_plot(result.data, "Зависимость cσ от времени")
        residual_plot(
            result.data.assign(T_pred=result.data["sigma_pred_pct"], error_celsius=result.data["c_sigma"] - result.data["sigma_pred_pct"]),
            "Остатки по cσ",
        )
    with st.expander("Подробная статистическая сводка"):
        st.text(result.model_summary)


def predict_temperature_engineering(params: dict[str, float], D: float, tau: float, c_sigma: float, G: float | None = None) -> float:
    inv_t = (
        params.get("const", 0.0)
        + params.get("ln_D", 0.0) * np.log(D)
        + params.get("ln_tau", 0.0) * np.log(tau)
        + params.get("ln_c_sigma", 0.0) * np.log(c_sigma)
    )
    if G is not None and "G" in params:
        inv_t += params.get("G", 0.0) * G
    if inv_t <= 0:
        raise ValueError("Базовая модель дала неположительное значение 1/T. Проверьте введенные параметры.")
    return 1.0 / inv_t - 273.15


def predict_temperature_improved(params: dict[str, float], D: float, tau: float, c_sigma: float, G: float | None = None) -> float:
    a2 = params.get("inv_T", np.nan)
    if not np.isfinite(a2) or abs(a2) < 1e-12:
        raise ValueError("В улучшенной модели коэффициент при 1/T слишком мал для устойчивого расчета.")

    numerator = (
        np.log(D)
        - params.get("const", 0.0)
        - params.get("ln_tau", 0.0) * np.log(tau)
        - params.get("ln_c_sigma", 0.0) * np.log(c_sigma)
    )
    if G is not None and "G" in params:
        numerator -= params.get("G", 0.0) * G

    inv_t = numerator / a2
    if inv_t <= 0:
        raise ValueError("Улучшенная модель дала неположительное значение 1/T. Проверьте введенные параметры.")
    return 1.0 / inv_t - 273.15


def predict_temperature_diameter_growth(params: dict[str, float], D: float, tau: float) -> float:
    a2 = params.get("inv_T", np.nan)
    if not np.isfinite(a2) or abs(a2) < 1e-12:
        raise ValueError("В модели роста диаметра коэффициент при 1/T слишком мал для устойчивого расчета.")
    inv_t = (np.log(D) - params.get("const", 0.0) - params.get("ln_tau", 0.0) * np.log(tau)) / a2
    if inv_t <= 0:
        raise ValueError("Модель роста диаметра дала неположительное значение 1/T.")
    return 1.0 / inv_t - 273.15


def build_cleaned_diameter_grain_results(prepared_df: pd.DataFrame, valid_grains: list[float]) -> dict[float, FitResult]:
    cleaned_results: dict[float, FitResult] = {}
    for grain in valid_grains:
        grain_df = prepared_df[prepared_df["G"] == grain].copy()
        if len(grain_df) < 7:
            continue
        try:
            result = fit_diameter_growth_model(grain_df, include_grain=False)
        except Exception:
            continue
        apply_key = f"applied_exclude_diameter_grain_{grain}"
        selected = st.session_state.get(apply_key, [])
        if selected:
            filtered = grain_df[~grain_df["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                try:
                    result = fit_diameter_growth_model(filtered, include_grain=False)
                except Exception:
                    pass
        cleaned_results[grain] = result
    return cleaned_results


def build_cleaned_sigma_grain_results(prepared_df: pd.DataFrame, valid_grains: list[float]) -> dict[float, FitResult]:
    cleaned_results: dict[float, FitResult] = {}
    for grain in valid_grains:
        grain_df = prepared_df[prepared_df["G"] == grain].copy()
        if len(grain_df) < 7:
            continue
        try:
            result = fit_anchor_saturation_model(grain_df, include_grain=False)
        except Exception:
            continue
        selected = st.session_state.get(f"exclude_sigma_grain_{grain}", [])
        if selected:
            filtered = grain_df[~grain_df["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                try:
                    result = fit_anchor_saturation_model(filtered, include_grain=False)
                except Exception:
                    pass
        cleaned_results[grain] = result
    return cleaned_results


def get_recommended_diameter_exclusions(prepared_df: pd.DataFrame, valid_grains: list[float]) -> dict[float, list[str]]:
    recommendations: dict[float, list[str]] = {}
    for grain in valid_grains:
        grain_df = prepared_df[prepared_df["G"] == grain].copy()
        if len(grain_df) < 7:
            continue
        try:
            result = fit_diameter_growth_model(grain_df, include_grain=False)
        except Exception:
            continue
        recommendations[grain] = result.outlier_recommendation["point_id"].astype(str).tolist()
    return recommendations


def fit_diameter_universal_grain_size_model(cleaned_results: dict[float, FitResult]) -> tuple[dict[str, float], pd.DataFrame, str]:
    rows: list[dict[str, float]] = []
    for grain, result in cleaned_results.items():
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        params = result.params.set_index("Параметр модели")["Значение"].to_dict()
        rows.append(
            {
                "G": float(grain),
                "grain_size_mm": grain_size,
                "ln_grain_size": float(np.log(grain_size)),
                "a": float(params.get("const", np.nan)),
                "b": float(params.get("ln_tau", np.nan)),
                "c": float(params.get("inv_T", np.nan)),
                "R²": float(result.metrics.get("R²", np.nan)),
            }
        )
    coeff_df = pd.DataFrame(rows).dropna()
    if len(coeff_df) < 3:
        raise ValueError("Для универсальной модели нужно минимум 3 очищенные зерновые модели с известным размером зерна.")

    X = sm.add_constant(coeff_df[["ln_grain_size"]])
    model_a = sm.OLS(coeff_df["a"], X).fit()
    model_c = sm.OLS(coeff_df["c"], X).fit()

    b_source_df = coeff_df.sort_values(by=["R²", "grain_size_mm"], ascending=[False, False]).copy()
    excluded_b_grain = None
    if len(b_source_df) >= 4:
        worst_row = b_source_df.sort_values(by=["R²", "grain_size_mm"], ascending=[True, False]).iloc[0]
        excluded_b_grain = float(worst_row["G"])
        b_source_df = b_source_df[b_source_df["G"] != excluded_b_grain].copy()
    b_const = float(b_source_df["b"].mean())

    params = {
        "alpha0": float(model_a.params.get("const", np.nan)),
        "alpha1": float(model_a.params.get("ln_grain_size", np.nan)),
        "b_const": b_const,
        "gamma0": float(model_c.params.get("const", np.nan)),
        "gamma1": float(model_c.params.get("ln_grain_size", np.nan)),
        "r2_a": float(model_a.rsquared),
        "r2_c": float(model_c.rsquared),
    }
    included_b_grains = ", ".join(str(int(g)) if float(g).is_integer() else str(g) for g in b_source_df["G"].tolist())
    excluded_b_text = "нет"
    if excluded_b_grain is not None:
        excluded_b_text = str(int(excluded_b_grain)) if excluded_b_grain.is_integer() else str(excluded_b_grain)
    summary_text = (
        "Метамодель коэффициентов очищенных зерновых моделей по размеру зерна.\n\n"
        f"a(dg)=alpha0+alpha1·ln(dg), R²={model_a.rsquared:.4f}\n"
        f"b=const={b_const:.8f}, использованы зерна: {included_b_grains}, исключено зерно: {excluded_b_text}\n"
        f"c(dg)=gamma0+gamma1·ln(dg), R²={model_c.rsquared:.4f}"
    )
    return params, coeff_df, summary_text


def fit_sigma_universal_grain_size_model(
    cleaned_results: dict[float, FitResult],
    variant: str = "full",
) -> tuple[dict[str, float], pd.DataFrame, str]:
    rows: list[dict[str, float]] = []
    for grain, result in cleaned_results.items():
        if float(grain) not in SIGMA_UNIVERSAL_GRAINS:
            continue
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        params = result.params.set_index("Параметр модели")["Значение"].to_dict()
        rows.append(
            {
                "G": float(grain),
                "grain_size_mm": grain_size,
                "ln_grain_size": float(np.log(grain_size)),
                "log_a": float(params.get("log_a", np.nan)),
                "p_exp": float(params.get("p_exp", np.nan)),
                "m_exp": float(params.get("m_exp", np.nan)),
                "R²": float(result.metrics.get("R²", np.nan)),
                "RMSE_sigma": float(sigma_metric_summary(result.data).get("RMSE по cσ, %", np.nan)),
            }
        )
    coeff_df = pd.DataFrame(rows).dropna()
    if len(coeff_df) < 3:
        raise ValueError("Для универсальной sigma-модели нужны минимум 3 очищенные зерновые модели с известным размером зерна.")

    X = sm.add_constant(coeff_df[["ln_grain_size"]])
    model_log_a = sm.OLS(coeff_df["log_a"], X).fit()

    weights = 1.0 / np.maximum(coeff_df["RMSE_sigma"].to_numpy(dtype=float), 1e-9)

    def aggregate_const(series: pd.Series, mode: str) -> float:
        values = series.to_numpy(dtype=float)
        if mode == "mean":
            return float(np.mean(values))
        if mode == "median":
            return float(np.median(values))
        if mode == "weighted":
            return float(np.average(values, weights=weights))
        raise ValueError(f"Неизвестный режим константы: {mode}")

    if variant == "full":
        model_p = sm.OLS(coeff_df["p_exp"], X).fit()
        model_m = sm.OLS(coeff_df["m_exp"], X).fit()
        beta0 = float(model_p.params.get("const", np.nan))
        beta1 = float(model_p.params.get("ln_grain_size", np.nan))
        gamma0 = float(model_m.params.get("const", np.nan))
        gamma1 = float(model_m.params.get("ln_grain_size", np.nan))
        variant_label = "A(dg), p(dg), m(dg)"
    elif variant == "m_const_mean":
        model_p = sm.OLS(coeff_df["p_exp"], X).fit()
        model_m = None
        beta0 = float(model_p.params.get("const", np.nan))
        beta1 = float(model_p.params.get("ln_grain_size", np.nan))
        gamma0 = aggregate_const(coeff_df["m_exp"], "mean")
        gamma1 = 0.0
        variant_label = "A(dg), p(dg), m=const (среднее)"
    elif variant == "m_const_median":
        model_p = sm.OLS(coeff_df["p_exp"], X).fit()
        model_m = None
        beta0 = float(model_p.params.get("const", np.nan))
        beta1 = float(model_p.params.get("ln_grain_size", np.nan))
        gamma0 = aggregate_const(coeff_df["m_exp"], "median")
        gamma1 = 0.0
        variant_label = "A(dg), p(dg), m=const (медиана)"
    elif variant == "m_const_weighted":
        model_p = sm.OLS(coeff_df["p_exp"], X).fit()
        model_m = None
        beta0 = float(model_p.params.get("const", np.nan))
        beta1 = float(model_p.params.get("ln_grain_size", np.nan))
        gamma0 = aggregate_const(coeff_df["m_exp"], "weighted")
        gamma1 = 0.0
        variant_label = "A(dg), p(dg), m=const (взвешенное)"
    else:
        raise ValueError(f"Неизвестный вариант sigma-метамодели: {variant}")

    params = {
        "alpha0": float(model_log_a.params.get("const", np.nan)),
        "alpha1": float(model_log_a.params.get("ln_grain_size", np.nan)),
        "beta0": beta0,
        "beta1": beta1,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "r2_log_a": float(model_log_a.rsquared),
        "r2_p": float(model_p.rsquared) if model_p is not None else np.nan,
        "r2_m": float(model_m.rsquared) if model_m is not None else np.nan,
        "variant": variant,
        "variant_label": variant_label,
    }
    p_text = f"p(dg)=beta0+beta1·ln(dg), R²={model_p.rsquared:.4f}" if model_p is not None and beta1 != 0.0 else f"p=const={beta0:.8f}"
    m_text = f"m(dg)=gamma0+gamma1·ln(dg), R²={model_m.rsquared:.4f}" if model_m is not None and gamma1 != 0.0 else f"m=const={gamma0:.8f}"
    included_grains = ", ".join(str(int(g)) if float(g).is_integer() else str(g) for g in coeff_df["G"].tolist())
    summary_text = (
        f"Метамодель коэффициентов очищенных sigma-моделей по размеру зерна. Использованы зерна: {included_grains}. Вариант: {variant_label}.\n\n"
        f"log(A)(dg)=alpha0+alpha1·ln(dg), R²={model_log_a.rsquared:.4f}\n"
        f"{p_text}\n"
        f"{m_text}\n\n"
        "Итоговая универсальная форма:\n"
        "cσ = A(dg) · τ^p(dg) · ((T - 550) / 350)^m(dg)"
    )
    return params, coeff_df, summary_text


def predict_temperature_diameter_universal(params: dict[str, float], D: float, tau: float, grain_size_mm: float) -> float:
    ln_g = np.log(grain_size_mm)
    a_val = params["alpha0"] + params["alpha1"] * ln_g
    b_val = params["b_const"]
    c_val = params["gamma0"] + params["gamma1"] * ln_g
    if not np.isfinite(c_val) or abs(c_val) < 1e-12:
        raise ValueError("Универсальная модель дала слишком малый коэффициент при 1/T.")
    inv_t = (np.log(D) - a_val - b_val * np.log(tau)) / c_val
    if not np.isfinite(inv_t) or inv_t <= 0:
        raise ValueError("Универсальная модель дала неположительное значение 1/T.")
    return float(1.0 / inv_t - 273.15)


def predict_temperature_sigma_universal(params: dict[str, float], tau: float, c_sigma: float, grain_size_mm: float) -> float:
    if tau <= 0:
        raise ValueError("Время наработки должно быть больше нуля.")
    if c_sigma <= 0:
        raise ValueError("Содержание сигма-фазы должно быть больше нуля.")
    ln_g = np.log(grain_size_mm)
    log_a = params["alpha0"] + params["alpha1"] * ln_g
    p_exp = params["beta0"] + params["beta1"] * ln_g
    m_exp = params["gamma0"] + params["gamma1"] * ln_g
    if not np.isfinite(m_exp) or abs(m_exp) < 1e-12:
        raise ValueError("Универсальная sigma-модель дала слишком малый показатель степени m.")
    denom = np.exp(log_a) * np.power(max(tau, 1e-12), p_exp)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Универсальная sigma-модель дала некорректный множитель A·τ^p.")
    temp_norm = np.power(max(c_sigma / denom, 1e-12), 1.0 / m_exp)
    return float(np.clip(550.0 + 350.0 * temp_norm, 550.0, 900.0))


def evaluate_sigma_universal_model(params: dict[str, float], cleaned_results: dict[float, FitResult]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for grain, result in cleaned_results.items():
        if float(grain) not in SIGMA_UNIVERSAL_GRAINS:
            continue
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        df = result.data.copy()
        df["T_pred_universal"] = df.apply(
            lambda row: predict_temperature_sigma_universal(params, float(row["tau"]), float(row["c_sigma"]), grain_size),
            axis=1,
        )
        rows.append(df)
    if not rows:
        raise ValueError("Нет данных для оценки универсальной sigma-модели.")
    eval_df = pd.concat(rows, ignore_index=True)
    errors = eval_df["T"] - eval_df["T_pred_universal"]
    return {
        "Количество точек": float(len(eval_df)),
        "R² по T": float(r2_score(eval_df["T"], eval_df["T_pred_universal"])) if len(eval_df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(eval_df["T"], eval_df["T_pred_universal"]))),
        "MAE по T, °C": float(mean_absolute_error(eval_df["T"], eval_df["T_pred_universal"])),
        "MAPE по T, %": float(np.mean(np.abs(errors) / np.maximum(np.abs(eval_df["T"]), 1e-9)) * 100.0),
    }


def evaluate_diameter_universal_model(params: dict[str, float], cleaned_results: dict[float, FitResult]) -> dict[str, float]:
    rows: list[pd.DataFrame] = []
    for grain, result in cleaned_results.items():
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        df = result.data.copy()
        df["T_pred_universal"] = df.apply(
            lambda row: predict_temperature_diameter_universal(params, float(row["D"]), float(row["tau"]), grain_size),
            axis=1,
        )
        rows.append(df)
    if not rows:
        raise ValueError("Нет данных для оценки универсальной модели диаметра.")
    eval_df = pd.concat(rows, ignore_index=True)
    errors = eval_df["T"] - eval_df["T_pred_universal"]
    return {
        "Количество точек": float(len(eval_df)),
        "R² по T": float(r2_score(eval_df["T"], eval_df["T_pred_universal"])) if len(eval_df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(eval_df["T"], eval_df["T_pred_universal"]))),
        "MAE по T, °C": float(mean_absolute_error(eval_df["T"], eval_df["T_pred_universal"])),
        "MAPE по T, %": float(np.mean(np.abs(errors) / np.maximum(np.abs(eval_df["T"]), 1e-9)) * 100.0),
    }


def parse_optional_float(value: str) -> float | None:
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def clear_sigma_when_diameter_entered() -> None:
    if str(st.session_state.get("universal_choice_d", "")).strip():
        st.session_state["universal_choice_sigma"] = ""


def clear_diameter_when_sigma_entered() -> None:
    if str(st.session_state.get("universal_choice_sigma", "")).strip():
        st.session_state["universal_choice_d"] = ""


def show_diameter_grain_block(result: FitResult, grain_value: float) -> None:
    st.subheader(f"Модель роста диаметра для номера зерна {grain_value}")
    metric_cards(result.metrics)
    st.subheader("Коэффициенты модели")
    st.dataframe(result.params, use_container_width=True, hide_index=True)
    st.caption(result.model_label)
    st.code(result.formula_text, language="text")

    st.subheader("Калькулятор температуры для этого номера зерна")
    calc_params = result.params.set_index("Параметр модели")["Значение"].to_dict()
    with st.form(key=f"diameter_grain_form_{grain_value}"):
        c1, c2 = st.columns(2)
        with c1:
            tau_value = st.number_input(
                f"Время наработки τ для модели диаметра, зерно {grain_value}",
                min_value=1.0,
                value=1000.0,
                step=1.0,
                format="%.0f",
                key=f"diameter_tau_{grain_value}",
            )
        with c2:
            d_value = st.number_input(
                f"Эквивалентный диаметр D для зерна {grain_value}",
                min_value=0.01,
                value=10.0,
                step=0.01,
                format="%.2f",
                key=f"diameter_D_{grain_value}",
            )
        submitted = st.form_submit_button("Рассчитать")
    if submitted:
        try:
            temp_value = predict_temperature_diameter_growth(calc_params, d_value, tau_value)
            st.metric("Расчетная температура по модели диаметра, °C", f"{temp_value:.4f}")
        except Exception as exc:
            st.error(f"Не удалось выполнить расчет по модели диаметра: {exc}")

    show_result_block(
        result,
        key_prefix=f"diameter_grain_{grain_value}",
        include_grain=False,
        fit_function=fit_diameter_growth_model,
        preselect_outliers=True,
        auto_apply_selected=False,
    )


def predict_temperature_anchor_saturation(
    params: dict[str, float], D: float, tau: float, c_sigma: float, G: float | None = None
) -> float:
    if c_sigma <= 0:
        raise ValueError("Для sigma-модели по зернам содержание сигма-фазы должно быть больше нуля.")

    if all(key in params for key in ["log_a", "p_exp", "m_exp"]):
        grain_params = params
    else:
        if G is None:
            raise ValueError("Для модели по отдельным зернам нужно указать номер зерна G.")
        grain_key = f"grain_{float(G)}_"
        grain_params = {k[len(grain_key):]: v for k, v in params.items() if k.startswith(grain_key)}
        if not grain_params:
            grain_key = f"grain_{int(round(float(G)))}_"
            grain_params = {k[len(grain_key):]: v for k, v in params.items() if k.startswith(grain_key)}
        if not grain_params:
            raise ValueError(f"Для номера зерна G={G} нет отдельной sigma-модели.")

    log_a = grain_params.get("log_a", np.nan)
    p_exp = grain_params.get("p_exp", np.nan)
    m_exp = grain_params.get("m_exp", np.nan)
    denom = np.exp(log_a) * np.power(max(tau, 1e-12), p_exp)
    if not np.isfinite(denom) or denom <= 0 or not np.isfinite(m_exp) or abs(m_exp) < 1e-12:
        raise ValueError("Параметры sigma-модели по зерну некорректны для обратного расчета.")
    temp_norm = np.power(max(c_sigma / denom, 1e-12), 1.0 / m_exp)
    return float(np.clip(550.0 + 350.0 * temp_norm, 550.0, 900.0))


def show_model_comparison(base_result: FitResult, improved_result: FitResult, anchor_result: FitResult) -> None:
    st.subheader("Сравнение моделей")
    metric_order = [
        "R²",
        "Скорректированный R²",
        "RMSE, °C",
        "MAE, °C",
        "MAPE, %",
        "Среднее отклонение, °C",
        "Стандартное отклонение ошибки, °C",
        "Максимальное отклонение, °C",
        "Стандартная ошибка регрессии",
        "Корреляция факт/модель",
        "Коэффициент достоверности аппроксимации, %",
    ]
    comparison_df = pd.DataFrame(
        {
            "Метрика": metric_order,
            "Базовая модель": [base_result.metrics.get(metric, np.nan) for metric in metric_order],
            "Улучшенная модель": [improved_result.metrics.get(metric, np.nan) for metric in metric_order],
            "Sigma-модель по зернам": [anchor_result.metrics.get(metric, np.nan) for metric in metric_order],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    anchor_compare = pd.DataFrame(
        [
            {
                "Модель": "Базовая",
                "Прогноз для реальной точки, °C": base_result.metrics.get("Прогноз для реальной точки, °C", np.nan),
                "Отклонение от диапазона 570–600 °C, °C": base_result.metrics.get(
                    "Отклонение реальной точки от диапазона, °C", np.nan
                ),
            },
            {
                "Модель": "Улучшенная",
                "Прогноз для реальной точки, °C": improved_result.metrics.get("Прогноз для реальной точки, °C", np.nan),
                "Отклонение от диапазона 570–600 °C, °C": improved_result.metrics.get(
                    "Отклонение реальной точки от диапазона, °C", np.nan
                ),
            },
            {
                "Модель": "Sigma-модель по зернам",
                "Прогноз для реальной точки, °C": anchor_result.metrics.get("Прогноз для реальной точки, °C", np.nan),
                "Отклонение от диапазона 570–600 °C, °C": anchor_result.metrics.get(
                    "Отклонение реальной точки от диапазона, °C", np.nan
                ),
            },
        ]
    )
    st.subheader("Проверка по важной реальной точке")
    st.dataframe(anchor_compare, use_container_width=True, hide_index=True)


def show_multi_calculator(base_result: FitResult, improved_result: FitResult, anchor_result: FitResult) -> None:
    st.subheader("Калькулятор температуры по моделям")
    st.caption("Введите параметры структуры и наработки, затем нажмите кнопку расчета.")

    with st.form(key="multi_model_calc_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tau_value = st.number_input("Время наработки τ", min_value=1.0, value=1000.0, step=1.0, format="%.0f")
        with c2:
            d_value = st.number_input("Эквивалентный диаметр D", min_value=0.01, value=10.0, step=0.01, format="%.2f")
        with c3:
            sigma_value = st.number_input("Содержание сигма-фазы cσ, %", min_value=0.01, value=1.0, step=0.01, format="%.2f")
        with c4:
            grain_value = st.number_input("Номер зерна G", min_value=1.0, value=8.0, step=1.0, format="%.0f")
        submitted = st.form_submit_button("Рассчитать")

    if not submitted:
        return

    base_params = base_result.params.set_index("Параметр модели")["Значение"].to_dict()
    improved_params = improved_result.params.set_index("Параметр модели")["Значение"].to_dict()
    anchor_params = anchor_result.params.set_index("Параметр модели")["Значение"].to_dict()

    try:
        base_temp = predict_temperature_engineering(base_params, d_value, tau_value, sigma_value, grain_value)
        improved_temp = predict_temperature_improved(improved_params, d_value, tau_value, sigma_value, grain_value)
        anchor_temp = predict_temperature_anchor_saturation(anchor_params, d_value, tau_value, sigma_value, grain_value)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Температура по базовой модели, °C", f"{base_temp:.4f}")
        with r2:
            st.metric("Температура по улучшенной модели, °C", f"{improved_temp:.4f}")
        with r3:
            st.metric("Температура по sigma-модели по зернам, °C", f"{anchor_temp:.4f}")

        calc_df = pd.DataFrame(
            [
                {"Модель": "Базовая", "Расчетная температура, °C": base_temp},
                {"Модель": "Улучшенная", "Расчетная температура, °C": improved_temp},
                {"Модель": "Sigma-модель по зернам", "Расчетная температура, °C": anchor_temp},
            ]
        )
        st.dataframe(calc_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Не удалось выполнить расчет: {exc}")


def render_universal_models_tab(prepared_df: pd.DataFrame, valid_grains: list[float]) -> None:
    st.subheader("Универсальные модели по размеру зерна")
    st.markdown(f"**Научное обоснование:** {SCIENTIFIC_UNIVERSAL_SIGMA_PARAGRAPH}")

    if not valid_grains:
        st.warning("Для построения универсальных моделей недостаточно зерновых наборов с минимум 7 точками.")
        return

    diameter_error_local = None
    sigma_error_local = None
    diameter_payload = None
    sigma_variants_payload = []
    selected_sigma_variant = None

    try:
        cleaned_diameter_results = build_cleaned_diameter_grain_results(prepared_df, valid_grains)
        universal_diameter_params, diameter_coeff_df, diameter_summary = fit_diameter_universal_grain_size_model(cleaned_diameter_results)
        diameter_eval = evaluate_diameter_universal_model(universal_diameter_params, cleaned_diameter_results)
        diameter_payload = {
            "params": universal_diameter_params,
            "coeff_df": diameter_coeff_df,
            "summary": diameter_summary,
            "eval": diameter_eval,
        }
    except Exception as exc:
        diameter_error_local = str(exc)

    try:
        cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
        params_item, coeff_df_item, summary_item = fit_sigma_universal_grain_size_model(cleaned_sigma_results, variant="m_const_median")
        eval_item = evaluate_sigma_universal_model(params_item, cleaned_sigma_results)
        selected_sigma_variant = {
            "key": "m_const_median",
            "title": "A(dg), p(dg), m=const (медиана)",
            "params": params_item,
            "coeff_df": coeff_df_item,
            "summary": summary_item,
            "eval": eval_item,
        }
    except Exception as exc:
        sigma_error_local = str(exc)

    quality_rows = []
    if diameter_payload is not None:
        quality_rows.append(
            {
                "Модель": "Универсальная модель диаметра",
                "Версия": "a(dg), b=const, c(dg)",
                "R² по T": diameter_payload["eval"]["R² по T"],
                "RMSE по T, °C": diameter_payload["eval"]["RMSE по T, °C"],
                "MAE по T, °C": diameter_payload["eval"]["MAE по T, °C"],
                "MAPE по T, %": diameter_payload["eval"]["MAPE по T, %"],
                "Количество точек": diameter_payload["eval"]["Количество точек"],
            }
        )
    if selected_sigma_variant is not None:
        quality_rows.append(
            {
                "Модель": "Универсальная sigma-модель",
                "Версия": selected_sigma_variant["title"],
                "R² по T": selected_sigma_variant["eval"]["R² по T"],
                "RMSE по T, °C": selected_sigma_variant["eval"]["RMSE по T, °C"],
                "MAE по T, °C": selected_sigma_variant["eval"]["MAE по T, °C"],
                "MAPE по T, %": selected_sigma_variant["eval"]["MAPE по T, %"],
                "Количество точек": selected_sigma_variant["eval"]["Количество точек"],
            }
        )
    if quality_rows:
        st.subheader("Сравнение качества универсальных моделей")
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Формула универсальной модели диаметра")
        if diameter_payload is None:
            st.error(f"Модель диаметра недоступна: {diameter_error_local}")
        else:
            p = diameter_payload["params"]
            st.code(
                "\n".join(
                    [
                        "ln(D) = a(dg) + b·ln(τ) + c(dg)·(1/T(K))",
                        f"a(dg) = {p['alpha0']:.8f} + ({p['alpha1']:.8f}) · ln(dg)",
                        f"b = {p['b_const']:.8f}",
                        f"c(dg) = {p['gamma0']:.8f} + ({p['gamma1']:.8f}) · ln(dg)",
                    ]
                ),
                language="text",
            )
            with st.expander("Сводка по универсальной модели диаметра"):
                st.text(diameter_payload["summary"])

    with col_right:
        st.subheader("Формула универсальной sigma-модели")
        if selected_sigma_variant is None:
            st.error(f"Sigma-модель недоступна: {sigma_error_local}")
        else:
            p = selected_sigma_variant["params"]
            st.code(
                "\n".join(
                    [
                        "cσ = A(dg) · τ^p(dg) · ((T - 550) / 350)^m(dg)",
                        f"log(A)(dg) = {p['alpha0']:.8f} + ({p['alpha1']:.8f}) · ln(dg)",
                        f"p(dg) = {p['beta0']:.8f} + ({p['beta1']:.8f}) · ln(dg)",
                        f"m(dg) = {p['gamma0']:.8f} + ({p['gamma1']:.8f}) · ln(dg)",
                    ]
                ),
                language="text",
            )
            with st.expander("Сводка по выбранной универсальной sigma-модели"):
                st.text(selected_sigma_variant["summary"])

    st.subheader("Калькулятор по универсальным моделям")
    st.caption("Можно ввести диаметр, процент σ-фазы или оба параметра сразу. Расчет запускается только по кнопке.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tau_value = st.number_input("Время наработки τ", min_value=1.0, value=1000.0, step=1.0, format="%.0f", key="universal_choice_tau")
    with c2:
        grain_number = st.selectbox("Номер зерна G", sorted(GRAIN_SIZE_MM.keys()), key="universal_choice_grain")
    with c3:
        st.text_input(
            "Эквивалентный диаметр D, мкм",
            key="universal_choice_d",
            placeholder="например, 7.25",
        )
    with c4:
        st.text_input(
            "Содержание σ-фазы, %",
            key="universal_choice_sigma",
            placeholder="например, 4.50",
        )

    if st.button("Рассчитать", key="universal_models_calculate"):
        try:
            diameter_value = parse_optional_float(st.session_state.get("universal_choice_d", ""))
            sigma_value = parse_optional_float(st.session_state.get("universal_choice_sigma", ""))
            if diameter_value is None and sigma_value is None:
                raise ValueError("Нужно заполнить хотя бы одно поле: диаметр и/или процент σ-фазы.")
            grain_size = GRAIN_SIZE_MM[float(grain_number)]
            result_rows = []
            if diameter_value is not None:
                if diameter_payload is None:
                    raise ValueError(diameter_error_local or "Универсальная модель диаметра недоступна.")
                temp_d = predict_temperature_diameter_universal(diameter_payload["params"], round(diameter_value, 2), tau_value, grain_size)
                result_rows.append({"Модель": "Универсальная модель диаметра", "Расчетная температура, °C": temp_d})
            if sigma_value is not None:
                if selected_sigma_variant is None:
                    raise ValueError(sigma_error_local or "Универсальная sigma-модель недоступна.")
                temp_sigma = predict_temperature_sigma_universal(selected_sigma_variant["params"], tau_value, round(sigma_value, 2), grain_size)
                result_rows.append({"Модель": "Универсальная sigma-модель", "Расчетная температура, °C": temp_sigma})

            if len(result_rows) == 1:
                row = result_rows[0]
                st.success(f"{row['Модель']}: {row['Расчетная температура, °C']:.4f} °C")
            else:
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.metric("Температура по универсальной модели диаметра, °C", f"{result_rows[0]['Расчетная температура, °C']:.4f}")
                with c_res2:
                    st.metric("Температура по универсальной sigma-модели, °C", f"{result_rows[1]['Расчетная температура, °C']:.4f}")
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Не удалось выполнить расчет: {exc}")


def show_result_block(
    result: FitResult,
    key_prefix: str = "main",
    include_grain: bool = True,
    fit_function=fit_engineering_model,
    preselect_outliers: bool = True,
    auto_apply_selected: bool = True,
) -> None:
    st.subheader("Показатели качества модели")
    metric_cards(result.metrics)

    st.subheader("Коэффициенты модели")
    st.dataframe(result.params, use_container_width=True, hide_index=True)

    st.caption(result.model_label)
    st.code(result.formula_text, language="text")

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
            default=outlier_labels if preselect_outliers else [],
            key=f"exclude_{key_prefix}",
        )
        effective_selected = list(selected)
        if not auto_apply_selected:
            apply_key = f"applied_exclude_{key_prefix}"
            c_apply, c_reset = st.columns(2)
            with c_apply:
                if st.button("Применить исключение выбранных точек", key=f"apply_{key_prefix}"):
                    st.session_state[apply_key] = list(selected)
            with c_reset:
                if st.button("Сбросить исключения", key=f"reset_{key_prefix}"):
                    st.session_state[apply_key] = []
            effective_selected = st.session_state.get(apply_key, [])
            if effective_selected:
                st.info(f"Сейчас реально исключено точек: {len(effective_selected)}")
        if effective_selected:
            filtered = result.data[~result.data["point_id"].astype(str).isin(effective_selected)].copy()
            if len(filtered) >= 7:
                st.info(f"Пересчет после исключения {len(effective_selected)} точек.")
                recalculated = fit_function(filtered, include_grain=include_grain)
                metric_cards(recalculated.metrics)
                st.dataframe(
                    recalculated.params,
                    use_container_width=True,
                    hide_index=True,
                )
                st.code(recalculated.formula_text, language="text")
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
    "Приложение поддерживает базовую инженерную модель и улучшенную физически ориентированную альтернативу. "
    "Во всех расчетах температура внутри формул переводится в Кельвины."
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

base_result = None
base_error = None
try:
    base_result = fit_engineering_model(prepared_df)
except Exception as exc:
    base_error = str(exc)

improved_result = None
improved_error = None
try:
    improved_result = fit_improved_model(prepared_df)
except Exception as exc:
    improved_error = str(exc)

diameter_result = None
diameter_error = None
try:
    diameter_result = fit_diameter_growth_model(prepared_df, include_grain=True)
except Exception as exc:
    diameter_error = str(exc)

anchor_result = None
anchor_error = None
try:
    anchor_result = fit_anchor_saturation_model(prepared_df)
except Exception as exc:
    anchor_error = str(exc)

def enrich_real_point_metrics(result: FitResult, predictor) -> None:
    params = result.params.set_index("Параметр модели")["Значение"].to_dict()
    try:
        temp = predictor(
            params,
            REAL_WORLD_POINT["D"],
            REAL_WORLD_POINT["tau"],
            REAL_WORLD_POINT["c_sigma"],
            REAL_WORLD_POINT["G"],
        )
        result.metrics["Прогноз для реальной точки, °C"] = float(temp)
        if REAL_WORLD_POINT["temp_min"] <= temp <= REAL_WORLD_POINT["temp_max"]:
            result.metrics["Отклонение реальной точки от диапазона, °C"] = 0.0
        elif temp < REAL_WORLD_POINT["temp_min"]:
            result.metrics["Отклонение реальной точки от диапазона, °C"] = REAL_WORLD_POINT["temp_min"] - temp
        else:
            result.metrics["Отклонение реальной точки от диапазона, °C"] = temp - REAL_WORLD_POINT["temp_max"]
    except Exception:
        result.metrics["Прогноз для реальной точки, °C"] = np.nan
        result.metrics["Отклонение реальной точки от диапазона, °C"] = np.nan


if base_result is not None:
    enrich_real_point_metrics(base_result, predict_temperature_engineering)
if improved_result is not None:
    enrich_real_point_metrics(improved_result, predict_temperature_improved)
if diameter_result is not None:
    enrich_real_point_metrics(diameter_result, lambda params, D, tau, c_sigma, G: predict_temperature_diameter_growth(params, D, tau))
if anchor_result is not None:
    enrich_real_point_metrics(anchor_result, predict_temperature_anchor_saturation)

main_tab, grain_tab, improved_tab, diameter_tab, anchor_tab, compare_tab, calculator_tab, universal_models_tab = st.tabs([
    "Общая модель",
    "Модели по номерам зерна",
    "Улучшенная модель",
    "Рост диаметра",
    "Простая sigma-модель",
    "Сравнение моделей",
    "Калькулятор",
    "Универсальные модели",
])

with main_tab:
    if base_result is not None:
        show_result_block(base_result, key_prefix="all", include_grain=True, fit_function=fit_engineering_model)
    else:
        st.error(f"Не удалось построить общую модель: {base_error}")

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
                grain_result = fit_engineering_model(grain_df, include_grain=False)
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
                    show_result_block(
                        grain_result,
                        key_prefix=f"grain_{grain}",
                        include_grain=False,
                        fit_function=fit_engineering_model,
                    )
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

with improved_tab:
    st.write(
        "Улучшенная модель из научного заключения: ln(D) = a0 + a1·ln(τ) + a2·(1/T) + a3·G + a4·ln(cσ). "
        "Для удобства ученого программа, как и раньше, пересчитывает из этой зависимости температуру и показывает все те же метрики, графики и слабые точки."
    )

    improved_main_tab, improved_grain_tab = st.tabs([
        "Общая улучшенная модель",
        "Улучшенные модели по номерам зерна",
    ])

    with improved_main_tab:
        try:
            if improved_result is None:
                raise ValueError(improved_error or "неизвестная ошибка")
            show_result_block(
                improved_result,
                key_prefix="improved_all",
                include_grain=True,
                fit_function=fit_improved_model,
            )
        except Exception as exc:
            st.error(f"Не удалось построить улучшенную модель: {exc}")

    with improved_grain_tab:
        improved_grain_scores: list[dict[str, float]] = []

        if not valid_grains:
            st.warning("Для отдельных номеров зерна пока недостаточно точек. Нужно минимум 7 точек на номер зерна.")
        else:
            selected_improved_grain = st.selectbox(
                "Выберите номер зерна для улучшенной модели",
                valid_grains,
            )
            for grain in valid_grains:
                grain_df = prepared_df[prepared_df["G"] == grain].copy()
                try:
                    grain_result = fit_improved_model(grain_df, include_grain=False)
                    improved_grain_scores.append(
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
                    if grain == selected_improved_grain:
                        show_result_block(
                            grain_result,
                            key_prefix=f"improved_grain_{grain}",
                            include_grain=False,
                            fit_function=fit_improved_model,
                        )
                except Exception:
                    continue

            if improved_grain_scores:
                st.subheader("Сравнение качества улучшенной модели по номерам зерна")
                score_df = pd.DataFrame(improved_grain_scores).sort_values(
                    by=["R²", "RMSE, °C", "MAPE, %"],
                    ascending=[False, True, True],
                )
                st.dataframe(score_df, use_container_width=True, hide_index=True)
                best_grain = score_df.iloc[0]
                st.info(
                    f"Лучше всего улучшенная модель выглядит для номера зерна {best_grain['Номер зерна']}: "
                    f"R²={best_grain['R²']:.4f}, RMSE={best_grain['RMSE, °C']:.4f} °C."
                )

with diameter_tab:
    st.write(
        "Модель роста диаметра тоже строится отдельно для каждого номера зерна. "
        "Для каждого G используется своя зависимость ln(D) = a + b·ln(τ) + c·(1/T), потому что скорость укрупнения сильно зависит от зерна."
    )
    diameter_grain_scores: list[dict[str, float]] = []
    if not valid_grains:
        st.warning("Для отдельных номеров зерна пока недостаточно точек. Нужно минимум 7 точек на номер зерна.")
    else:
        local_tab, universal_tab = st.tabs(["Модели по отдельным зернам", "Универсальная модель"])
        with local_tab:
            cleaned_diameter_results = build_cleaned_diameter_grain_results(prepared_df, valid_grains)
            selected_diameter_grain = st.selectbox("Выберите номер зерна для модели диаметра", valid_grains)
            for grain in valid_grains:
                grain_df = prepared_df[prepared_df["G"] == grain].copy()
                try:
                    grain_result = cleaned_diameter_results.get(grain) or fit_diameter_growth_model(grain_df, include_grain=False)
                    diameter_grain_scores.append(
                        {
                            "Номер зерна": grain,
                            "Количество точек": grain_result.metrics["Количество точек"],
                            "R²": grain_result.metrics["R²"],
                            "RMSE, °C": grain_result.metrics["RMSE, °C"],
                            "MAE, °C": grain_result.metrics["MAE, °C"],
                            "MAPE, %": grain_result.metrics["MAPE, %"],
                        }
                    )
                    if grain == selected_diameter_grain:
                        show_diameter_grain_block(grain_result, grain)
                except Exception:
                    continue

            if diameter_grain_scores:
                st.subheader("Сравнение моделей роста диаметра по номерам зерна")
                diameter_score_df = pd.DataFrame(diameter_grain_scores).sort_values(
                    by=["R²", "RMSE, °C", "MAPE, %"],
                    ascending=[False, True, True],
                )
                st.dataframe(diameter_score_df, use_container_width=True, hide_index=True)
                best_diameter_grain = diameter_score_df.iloc[0]
                st.info(
                    f"Лучше всего модель роста диаметра сейчас выглядит для номера зерна {best_diameter_grain['Номер зерна']}: "
                    f"R²={best_diameter_grain['R²']:.4f}, RMSE={best_diameter_grain['RMSE, °C']:.4f} °C."
                )

        with universal_tab:
            recommended_exclusions = get_recommended_diameter_exclusions(prepared_df, valid_grains)
            c_apply_all, c_reset_all = st.columns(2)
            with c_apply_all:
                if st.button("Применить все рекомендованные исключения по всем зернам", key="apply_all_diameter_exclusions"):
                    for grain, labels in recommended_exclusions.items():
                        st.session_state[f"applied_exclude_diameter_grain_{grain}"] = list(labels)
            with c_reset_all:
                if st.button("Сбросить все исключения по росту диаметра", key="reset_all_diameter_exclusions"):
                    for grain in valid_grains:
                        st.session_state[f"applied_exclude_diameter_grain_{grain}"] = []

            active_rows = []
            for grain in valid_grains:
                active_rows.append(
                    {
                        "Номер зерна": grain,
                        "Рекомендовано исключить": len(recommended_exclusions.get(grain, [])),
                        "Сейчас исключено": len(st.session_state.get(f"applied_exclude_diameter_grain_{grain}", [])),
                    }
                )
            st.dataframe(pd.DataFrame(active_rows), use_container_width=True, hide_index=True)

            cleaned_diameter_results = build_cleaned_diameter_grain_results(prepared_df, valid_grains)
            st.subheader("Универсальная модель по размеру зерна")
            try:
                universal_params, coeff_df, universal_summary = fit_diameter_universal_grain_size_model(cleaned_diameter_results)
                formula_text = (
                    "ln(D) = a(dg) + b·ln(τ) + c(dg)·(1/T(K))\n"
                    "a(dg) = alpha0 + alpha1·ln(dg)\n"
                    "b = const\n"
                    "c(dg) = gamma0 + gamma1·ln(dg)\n"
                    f"alpha0 = {universal_params['alpha0']:.8f}, alpha1 = {universal_params['alpha1']:.8f}\n"
                    f"b = {universal_params['b_const']:.8f}\n"
                    f"gamma0 = {universal_params['gamma0']:.8f}, gamma1 = {universal_params['gamma1']:.8f}"
                )
                st.code(formula_text, language="text")
                st.dataframe(coeff_df, use_container_width=True, hide_index=True)
                with st.expander("Сводка по универсальной модели"):
                    st.text(universal_summary)

                st.subheader("Калькулятор температуры по универсальной модели")
                with st.form(key="diameter_universal_form"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        tau_value = st.number_input("Время наработки τ для универсальной модели", min_value=1.0, value=1000.0, step=1.0, format="%.0f", key="diameter_universal_tau")
                    with c2:
                        d_value = st.number_input("Эквивалентный диаметр D для универсальной модели", min_value=0.01, value=10.0, step=0.01, format="%.2f", key="diameter_universal_D")
                    with c3:
                        grain_number = st.selectbox("Номер зерна для универсальной модели", sorted(GRAIN_SIZE_MM.keys()), key="diameter_universal_grain")
                    submitted = st.form_submit_button("Рассчитать")
                if submitted:
                    try:
                        temp_value = predict_temperature_diameter_universal(universal_params, d_value, tau_value, GRAIN_SIZE_MM[float(grain_number)])
                        st.metric("Расчетная температура по универсальной модели, °C", f"{temp_value:.4f}")
                    except Exception as exc:
                        st.error(f"Не удалось выполнить расчет по универсальной модели: {exc}")
            except Exception as exc:
                st.error(f"Не удалось собрать универсальную модель по размеру зерна: {exc}")

with anchor_tab:
    st.write(
        "Здесь показаны отдельные прямые степенные модели содержания сигма-фазы для каждого номера зерна, без logit-преобразования. "
        "Для каждого G зависимость строится отдельно по температуре и времени, с собственными коэффициентами, графиками и оценкой качества."
    )
    sigma_grain_scores: list[dict[str, float]] = []
    if not valid_grains:
        st.warning("Для отдельных номеров зерна пока недостаточно точек. Нужно минимум 7 точек на номер зерна.")
    else:
        selected_sigma_grain = st.selectbox("Выберите номер зерна для sigma-модели", valid_grains)
        for grain in valid_grains:
            grain_df = prepared_df[prepared_df["G"] == grain].copy()
            try:
                grain_result = fit_anchor_saturation_model(grain_df, include_grain=False)
                sigma_metrics = sigma_metric_summary(grain_result.data)
                sigma_grain_scores.append(
                    {
                        "Номер зерна": grain,
                        **sigma_metrics,
                    }
                )
                if grain == selected_sigma_grain:
                    show_sigma_grain_block(grain_result, grain)
            except Exception:
                continue

        if sigma_grain_scores:
            st.subheader("Сравнение sigma-моделей по номерам зерна")
            sigma_score_df = pd.DataFrame(sigma_grain_scores).sort_values(
                by=["R² по cσ", "RMSE по cσ, %", "MAPE по cσ, %"],
                ascending=[False, True, True],
            )
            st.dataframe(sigma_score_df, use_container_width=True, hide_index=True)
            best_sigma_grain = sigma_score_df.iloc[0]
            st.info(
                f"Лучше всего sigma-модель сейчас выглядит для номера зерна {best_sigma_grain['Номер зерна']}: "
                f"R² по cσ={best_sigma_grain['R² по cσ']:.4f}, RMSE по cσ={best_sigma_grain['RMSE по cσ, %']:.4f} %"
            )

        st.subheader("Универсальная sigma-модель по размеру зерна")
        st.write(
            "Подход повторяет универсальную модель роста диаметра: сначала строятся отдельные sigma-модели "
            "для каждого номера зерна, затем коэффициенты log(A) и p выражаются через логарифм среднего размера зерна. "
            "В общую sigma-модель включены только зерна 8, 9 и 10, а коэффициент m фиксирован как константа по медиане, "
            "потому что такой вариант устойчивее к шуму и не даёт зерну 5 портить общую зависимость."
        )
        try:
            cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
            selected_params, sigma_coeff_df, sigma_summary = fit_sigma_universal_grain_size_model(cleaned_sigma_results, variant="m_const_median")
            sigma_eval = evaluate_sigma_universal_model(selected_params, cleaned_sigma_results)

            coeff_view = sigma_coeff_df[["G", "grain_size_mm", "log_a", "p_exp", "m_exp", "R²", "RMSE_sigma"]].copy()
            coeff_view = coeff_view.rename(
                columns={
                    "grain_size_mm": "Размер зерна, мм",
                    "log_a": "log(A)",
                    "p_exp": "p",
                    "m_exp": "m",
                    "R²": "R² по T",
                    "RMSE_sigma": "RMSE по cσ, %",
                }
            )
            st.dataframe(coeff_view, use_container_width=True, hide_index=True)

            st.info(
                f"Для общей sigma-модели используются только зерна 8, 9 и 10. Текущий вариант: m = const по медиане. "
                f"RMSE={sigma_eval['RMSE по T, °C']:.4f} °C, R²={sigma_eval['R² по T']:.4f}."
            )

            st.code(
                "\n".join(
                    [
                        f"log(A)(dg) = {selected_params['alpha0']:.8f} + ({selected_params['alpha1']:.8f}) · ln(dg)",
                        f"p(dg) = {selected_params['beta0']:.8f} + ({selected_params['beta1']:.8f}) · ln(dg)",
                        f"m = {selected_params['gamma0']:.8f}",
                        "",
                        "cσ = A(dg) · τ^p(dg) · ((T - 550) / 350)^m",
                    ]
                ),
                language="text",
            )
            meta_quality_df = pd.DataFrame(
                [
                    {
                        "R² для log(A)(dg)": selected_params["r2_log_a"],
                        "R² для p(dg)": selected_params["r2_p"],
                        "R² по T": sigma_eval["R² по T"],
                        "RMSE по T, °C": sigma_eval["RMSE по T, °C"],
                        "Количество зерновых моделей": float(len(sigma_coeff_df)),
                    }
                ]
            )
            st.dataframe(meta_quality_df, use_container_width=True, hide_index=True)
            with st.expander("Сводка по универсальной sigma-модели"):
                st.text(sigma_summary)

            st.subheader("Калькулятор температуры по универсальной sigma-модели")
            with st.form(key="sigma_universal_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    sigma_tau_value = st.number_input(
                        "Время наработки τ для универсальной sigma-модели",
                        min_value=1.0,
                        value=1000.0,
                        step=1.0,
                        format="%.0f",
                        key="sigma_universal_tau",
                    )
                with c2:
                    sigma_value = st.number_input(
                        "Содержание сигма-фазы cσ для универсальной модели, %",
                        min_value=0.01,
                        value=1.0,
                        step=0.01,
                        format="%.2f",
                        key="sigma_universal_sigma",
                    )
                with c3:
                    sigma_grain_number = st.selectbox(
                        "Номер зерна для универсальной sigma-модели",
                        sorted(GRAIN_SIZE_MM.keys()),
                        key="sigma_universal_grain",
                    )
                submitted = st.form_submit_button("Рассчитать")
            if submitted:
                try:
                    sigma_temp_value = predict_temperature_sigma_universal(
                        selected_params,
                        sigma_tau_value,
                        sigma_value,
                        GRAIN_SIZE_MM[float(sigma_grain_number)],
                    )
                    st.metric("Расчетная температура по универсальной sigma-модели, °C", f"{sigma_temp_value:.4f}")
                except Exception as exc:
                    st.error(f"Не удалось выполнить расчет по универсальной sigma-модели: {exc}")
        except Exception as exc:
            st.error(f"Не удалось собрать универсальную sigma-модель по размеру зерна: {exc}")

with compare_tab:
    if base_result is None:
        st.error(f"Базовая модель недоступна для сравнения: {base_error}")
    elif improved_result is None:
        st.error(f"Улучшенная модель недоступна для сравнения: {improved_error}")
    elif anchor_result is None:
        st.error(f"Sigma-модель по отдельным зернам недоступна для сравнения: {anchor_error}")
    else:
        show_model_comparison(base_result, improved_result, anchor_result)

        grain_compare_rows: list[dict[str, float]] = []
        for grain in valid_grains:
            grain_df = prepared_df[prepared_df["G"] == grain].copy()
            try:
                base_grain_result = fit_engineering_model(grain_df, include_grain=False)
                improved_grain_result = fit_improved_model(grain_df, include_grain=False)
                anchor_grain_result = fit_anchor_saturation_model(grain_df, include_grain=False)
                grain_compare_rows.append(
                    {
                        "Номер зерна": grain,
                        "R² базовая": base_grain_result.metrics["R²"],
                        "R² улучшенная": improved_grain_result.metrics["R²"],
                        "R² sigma по зерну": anchor_grain_result.metrics["R²"]},{
                        "RMSE базовая, °C": base_grain_result.metrics["RMSE, °C"],
                        "RMSE улучшенная, °C": improved_grain_result.metrics["RMSE, °C"],
                        "RMSE sigma по зерну, °C": anchor_grain_result.metrics["RMSE, °C"]},{
                        "MAPE базовая, %": base_grain_result.metrics["MAPE, %"],
                        "MAPE улучшенная, %": improved_grain_result.metrics["MAPE, %"],
                        "MAPE sigma по зерну, %": anchor_grain_result.metrics["MAPE, %"]},{
                    }
                )
            except Exception:
                continue

        if grain_compare_rows:
            st.subheader("Сравнение моделей по номерам зерна")
            grain_compare_df = pd.DataFrame(grain_compare_rows)
            st.dataframe(grain_compare_df, use_container_width=True, hide_index=True)

with calculator_tab:
    if base_result is None:
        st.error(f"Калькулятор базовой модели недоступен: {base_error}")
    elif improved_result is None:
        st.error(f"Калькулятор улучшенной модели недоступен: {improved_error}")
    elif anchor_result is None:
        st.error(f"Калькулятор sigma-модели по отдельным зернам недоступен: {anchor_error}")
    else:
        show_multi_calculator(base_result, improved_result, anchor_result)

with universal_models_tab:
    render_universal_models_tab(prepared_df, valid_grains)
