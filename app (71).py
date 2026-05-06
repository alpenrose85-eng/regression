from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
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
    "sigma_diameter",
    "диаметр сигмы",
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
    "sigma_pct",
    "процент сигмы",
    "сигма",
    "содержание сигма-фазы",
    "c sigma",
    "c_sigma_pct",
    "c sigma pct",
]
ID_ALIASES = ["id", "sample", "sample_id", "образец", "точка"]
ASSUMED_TEMP_ALIASES = [
    "предполагаемая температура",
    "предполагаемая_t",
    "предполагаемая t",
    "расчетная температура",
    "ожидаемая температура",
    "ожидаемая t",
    "target temperature",
    "target_temperature",
    "assumed temperature",
    "assumed_temperature",
    "expected temperature",
    "expected_temperature",
    "t_expected",
    "t_assumed",
    "t_target",
]

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

SIGMA_UNIVERSAL_GRAINS = [3.0, 5.0, 8.0, 9.0, 10.0]

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


def prepare_calibration_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    mapping = {
        "D": find_column(df.columns, D_ALIASES),
        "tau": find_column(df.columns, TAU_ALIASES),
        "G": find_column(df.columns, GRAIN_ALIASES),
        "c_sigma": find_column(df.columns, SIGMA_ALIASES),
        "T_assumed": find_column(df.columns, ASSUMED_TEMP_ALIASES),
        "point_id": find_column(df.columns, ID_ALIASES),
    }

    required_missing = [key for key in ["D", "tau", "G", "c_sigma", "T_assumed"] if mapping[key] is None]
    if required_missing:
        raise ValueError(
            "Не удалось автоматически распознать обязательные столбцы для калибровки: "
            + ", ".join(required_missing)
            + ". Нужны столбцы со временем, диаметром, номером зерна, процентом sigma-фазы и предполагаемой температурой."
        )

    prepared = pd.DataFrame(
        {
            "D": pd.to_numeric(df[mapping["D"]], errors="coerce"),
            "tau": pd.to_numeric(df[mapping["tau"]], errors="coerce"),
            "G": pd.to_numeric(df[mapping["G"]], errors="coerce"),
            "c_sigma": pd.to_numeric(df[mapping["c_sigma"]], errors="coerce"),
            "T_assumed": pd.to_numeric(df[mapping["T_assumed"]], errors="coerce"),
        }
    )

    if mapping["point_id"] is not None:
        prepared["point_id"] = df[mapping["point_id"]].astype(str)
    else:
        prepared["point_id"] = [f"Калибровка {i + 1}" for i in range(len(prepared))]

    extra_columns = [col for col in df.columns if col not in set(mapping.values()) - {None}]
    for col in extra_columns:
        prepared[col] = df[col]

    prepared = prepared.dropna(subset=["D", "tau", "G", "c_sigma", "T_assumed"]).copy()
    prepared = prepared[(prepared["D"] > 0) & (prepared["tau"] > 0) & (prepared["c_sigma"] > 0)].copy()
    prepared = prepared[prepared["G"] > 0].copy()
    prepared = prepared[prepared["T_assumed"] > -273.15].copy()

    if prepared.empty:
        raise ValueError("После очистки не осталось корректных строк для калибровки. Проверьте данные и единицы измерения.")

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


def build_calibration_template_workbook() -> bytes:
    template_df = pd.DataFrame(
        [
            {"point_id": "К1", "tau": 1000, "D": 4.2, "G": 8, "c_sigma": 3.5, "предполагаемая температура": 620},
            {"point_id": "К2", "tau": 5000, "D": 7.1, "G": 9, "c_sigma": 8.2, "предполагаемая температура": 660},
            {"point_id": "К3", "tau": 12000, "D": 9.4, "G": 10, "c_sigma": 12.6, "предполагаемая температура": 690},
        ]
    )
    help_df = pd.DataFrame(
        {
            "Поле": ["point_id", "tau", "D", "G", "c_sigma", "предполагаемая температура"],
            "Описание": [
                "Идентификатор точки (необязательно)",
                "Время / наработка",
                "Эквивалентный диаметр sigma",
                "Номер зерна",
                "Процент sigma-фазы",
                "Температура, с которой сравниваем модели",
            ],
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Калибровка")
        help_df.to_excel(writer, index=False, sheet_name="Описание")
    return buffer.getvalue()


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


def sigma_temperature_points_plot(df: pd.DataFrame, title: str) -> None:
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


def show_sigma_grain_block(result: FitResult, grain_value: float, grain_df: pd.DataFrame) -> None:
    apply_key = f"applied_exclude_sigma_grain_{grain_value}"
    widget_key = f"exclude_sigma_grain_{grain_value}"
    source_point_ids = grain_df["point_id"].astype(str).tolist()
    recommended_ids = result.outlier_recommendation["point_id"].astype(str).tolist()

    if apply_key not in st.session_state:
        st.session_state[apply_key] = []
    pending_sync_key = f"pending_sync_sigma_grain_{grain_value}"
    desired_selection = [pid for pid in st.session_state.get(apply_key, []) if pid in source_point_ids]
    if widget_key not in st.session_state:
        st.session_state[widget_key] = list(desired_selection)
    elif st.session_state.get(pending_sync_key):
        st.session_state[widget_key] = list(desired_selection)
        st.session_state[pending_sync_key] = False

    active_result = result
    effective_selected = [pid for pid in st.session_state.get(apply_key, []) if pid in source_point_ids]
    if effective_selected:
        filtered = grain_df[~grain_df["point_id"].astype(str).isin(effective_selected)].copy()
        if len(filtered) >= 7:
            try:
                active_result = fit_anchor_saturation_model(filtered, include_grain=False)
            except Exception:
                active_result = result

    st.subheader(f"Sigma-модель для номера зерна {grain_value}")
    st.caption("Сначала показана предсказательность модели по температуре, ниже — качество прямой подгонки по содержанию сигма-фазы.")

    with st.expander("Исключение точек для этого зерна", expanded=True):
        st.caption("Здесь сохраняются именно ваши применённые исключения. Рекомендованные точки можно добавить в выбор, но они не заменяют ваш список автоматически.")
        c_apply_rec, c_apply_user, c_reset = st.columns(3)
        with c_apply_rec:
            if st.button("Добавить рекомендованные", key=f"apply_recommended_sigma_grain_{grain_value}"):
                merged = sorted(set(st.session_state.get(widget_key, [])) | set(recommended_ids))
                st.session_state[widget_key] = merged
                st.rerun()
        with c_apply_user:
            if st.button("Применить выбранные точки", key=f"apply_sigma_grain_{grain_value}"):
                st.session_state[apply_key] = [pid for pid in st.session_state.get(widget_key, []) if pid in source_point_ids]
                st.session_state[pending_sync_key] = True
                st.rerun()
        with c_reset:
            if st.button("Сбросить исключения", key=f"reset_sigma_grain_{grain_value}"):
                st.session_state[apply_key] = []
                st.session_state[pending_sync_key] = True
                st.rerun()

        st.multiselect(
            "Какие точки исключить из sigma-модели",
            options=source_point_ids,
            key=widget_key,
            help="Сначала выберите точки, затем нажмите «Применить выбранные точки».",
        )
        st.write(f"Рекомендовано системой: {len(recommended_ids)}")
        st.write(f"Сейчас реально исключено: {len(effective_selected)}")
        if effective_selected:
            st.caption("Исключены точки: " + ", ".join(effective_selected))
        if len(source_point_ids) - len(effective_selected) < 7:
            st.error("После текущего исключения останется меньше 7 точек. Такая подгонка не будет использоваться.")

    st.subheader("Качество предсказания температуры")
    metric_cards(temperature_metric_summary(active_result.data))
    st.subheader("Качество подгонки по содержанию сигма-фазы")
    sigma_metrics = sigma_metric_summary(active_result.data)
    metric_cards(sigma_metrics)
    st.subheader("Коэффициенты модели")
    st.dataframe(active_result.params, use_container_width=True, hide_index=True)
    st.caption(active_result.model_label)
    st.code(active_result.formula_text, language="text")

    st.subheader("Калькулятор температуры для этого номера зерна")
    calc_params = active_result.params.set_index("Параметр модели")["Значение"].to_dict()
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

    st.subheader("Таблица по точкам")
    sigma_view = active_result.data[["point_id", "T", "tau", "G", "c_sigma", "sigma_pred_pct", "T_pred", "error_celsius"]].copy()
    sigma_view["Ошибка по cσ, %"] = sigma_view["c_sigma"] - sigma_view["sigma_pred_pct"]
    st.dataframe(sigma_view, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        sigma_scatter_fact_vs_pred(active_result.data, "Эксперимент vs модель по cσ")
        sigma_vs_temperature_plot(active_result.data, "Зависимость cσ от температуры")
    with c2:
        sigma_vs_time_plot(active_result.data, "Зависимость cσ от времени")
        residual_plot(
            active_result.data.assign(T_pred=active_result.data["sigma_pred_pct"], error_celsius=active_result.data["c_sigma"] - active_result.data["sigma_pred_pct"]),
            "Остатки по cσ",
        )
    sigma_temperature_points_plot(active_result.data, "Обратный расчет температуры: модель и экспериментальные точки")
    with st.expander("Подробная статистическая сводка"):
        st.text(active_result.model_summary)


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


def predict_temperature_diameter_grain_model(
    params: dict[str, float], D: float, tau: float, G: float | None = None
) -> float:
    if all(key in params for key in ["const", "ln_tau", "inv_T"]):
        grain_params = params
    else:
        if G is None:
            raise ValueError("Для модели роста диаметра по зернам нужно указать номер зерна G.")
        grain_key = f"grain_{float(G)}_"
        grain_params = {k[len(grain_key):]: v for k, v in params.items() if k.startswith(grain_key)}
        if not grain_params:
            grain_key = f"grain_{int(round(float(G)))}_"
            grain_params = {k[len(grain_key):]: v for k, v in params.items() if k.startswith(grain_key)}
        if not grain_params:
            raise ValueError(f"Для номера зерна G={G} нет отдельной модели роста диаметра.")

    return predict_temperature_diameter_growth(grain_params, D, tau)


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
        selected = st.session_state.get(f"applied_exclude_sigma_grain_{grain}", [])
        if selected:
            filtered = grain_df[~grain_df["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                try:
                    result = fit_anchor_saturation_model(filtered, include_grain=False)
                except Exception:
                    pass
        cleaned_results[grain] = result
    return cleaned_results


def get_recommended_sigma_exclusions(prepared_df: pd.DataFrame, valid_grains: list[float]) -> dict[float, list[str]]:
    recommendations: dict[float, list[str]] = {}
    for grain in valid_grains:
        grain_df = prepared_df[prepared_df["G"] == grain].copy()
        if len(grain_df) < 7:
            continue
        try:
            result = fit_anchor_saturation_model(grain_df, include_grain=False)
        except Exception:
            continue
        recommendations[grain] = result.outlier_recommendation["point_id"].astype(str).tolist()
    return recommendations


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


def fit_diameter_universal_grain_size_model(
    cleaned_results: dict[float, FitResult],
    variant: str = "quadratic_full",
) -> tuple[dict[str, float], pd.DataFrame, str]:
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

    ln_g = coeff_df["ln_grain_size"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(coeff_df)), ln_g, ln_g ** 2])
    model_a = sm.OLS(coeff_df["a"], X).fit()
    model_b = sm.OLS(coeff_df["b"], X).fit()
    model_c = sm.OLS(coeff_df["c"], X).fit()
    if variant != "quadratic_full":
        raise ValueError(f"Неизвестный вариант универсальной модели диаметра: {variant}")

    params = {
        "alpha0": 12.63978238,
        "alpha1": 5.57485560,
        "alpha2": 0.81060890,
        "beta0": 0.36498964,
        "beta1": 0.14874239,
        "beta2": 0.02449260,
        "b_const": np.nan,
        "gamma0": -13874.19081347,
        "gamma1": -6019.06457877,
        "gamma2": -868.42563050,
        "r2_a": float(model_a.rsquared),
        "r2_b": float(model_b.rsquared),
        "r2_c": float(model_c.rsquared),
        "variant": "quadratic_full",
        "variant_label": "a(dg), b(dg), c(dg) = u0 + u1·ln(dg) + u2·[ln(dg)]²",
        "variant_title": "Диаметр: a(dg), b(dg), c(dg)",
    }
    included_grains = ", ".join(str(int(g)) if float(g).is_integer() else str(g) for g in coeff_df["G"].tolist())
    summary_text = (
        "Метамодель коэффициентов очищенных зерновых моделей по размеру зерна.\n\n"
        f"a(dg)=alpha0+alpha1·ln(dg)+alpha2·[ln(dg)]², R²={model_a.rsquared:.4f}\n"
        f"b(dg)=beta0+beta1·ln(dg)+beta2·[ln(dg)]², R²={model_b.rsquared:.4f}\n"
        f"c(dg)=gamma0+gamma1·ln(dg)+gamma2·[ln(dg)]², R²={model_c.rsquared:.4f}\n"
        f"Использованы все зерна: {included_grains}"
    )
    return params, coeff_df, summary_text


def analyze_coefficient_forms(coeff_df: pd.DataFrame, coeff_name: str) -> pd.DataFrame:
    source = coeff_df[["G", "grain_size_mm", "ln_grain_size", coeff_name]].dropna().copy()
    y = source[coeff_name].to_numpy(dtype=float)
    dg = source["grain_size_mm"].to_numpy(dtype=float)
    ln_dg = source["ln_grain_size"].to_numpy(dtype=float)

    candidates = [
        (
            "u0 + u1·ln(dg)",
            np.column_stack([np.ones(len(source)), ln_dg]),
            ["u0", "u1"],
        ),
        (
            "u0 + u1·dg",
            np.column_stack([np.ones(len(source)), dg]),
            ["u0", "u1"],
        ),
        (
            "u0 + u1·(1/dg)",
            np.column_stack([np.ones(len(source)), 1.0 / dg]),
            ["u0", "u1"],
        ),
        (
            "u0 + u1·ln(dg) + u2·[ln(dg)]²",
            np.column_stack([np.ones(len(source)), ln_dg, ln_dg ** 2]),
            ["u0", "u1", "u2"],
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for label, X, names in candidates:
        model = sm.OLS(y, X).fit()
        row: dict[str, float | str] = {
            "Коэффициент": coeff_name,
            "Форма": label,
            "R²": float(model.rsquared),
        }
        params_arr = np.asarray(model.params, dtype=float)
        for idx, name in enumerate(names):
            row[name] = float(params_arr[idx])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["R²", "Форма"], ascending=[False, True]).reset_index(drop=True)


def fit_sigma_universal_grain_size_model(
    cleaned_results: dict[float, FitResult],
    variant: str = "median_constants",
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

    ln_g = coeff_df["ln_grain_size"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(coeff_df)), ln_g, ln_g ** 2])
    model_log_a = sm.OLS(coeff_df["log_a"], X).fit()

    params_a = np.asarray(model_log_a.params, dtype=float)

    if variant != "median_constants":
        raise ValueError(f"Неизвестный вариант универсальной sigma-модели: {variant}")
    p_const = float(coeff_df["p_exp"].median())
    m_const = float(coeff_df["m_exp"].median())
    variant_label = "log(A)(dg) = u0 + u1·ln(dg) + u2·[ln(dg)]²; p и m = median"
    variant_title = "log(A)(dg), p и m = медианы"

    params = {
        "alpha0": float(params_a[0]),
        "alpha1": float(params_a[1]),
        "alpha2": float(params_a[2]),
        "p_const": p_const,
        "m_const": m_const,
        "r2_log_a": float(model_log_a.rsquared),
        "variant": variant,
        "variant_label": variant_label,
        "variant_title": variant_title,
    }
    included_grains = ", ".join(str(int(g)) if float(g).is_integer() else str(g) for g in coeff_df["G"].tolist())
    summary_text = (
        f"Метамодель коэффициентов очищенных sigma-моделей по размеру зерна. Использованы зерна: {included_grains}.\n\n"
        f"log(A)(dg)=alpha0+alpha1·ln(dg)+alpha2·[ln(dg)]², R²={model_log_a.rsquared:.4f}\n"
        f"p={p_const:.10f} (константа)\n"
        f"m={m_const:.10f} (константа)\n\n"
        "Итоговая универсальная форма:\n"
        "cσ = A(dg) · τ^p · ((T - 550) / 350)^m"
    )
    return params, coeff_df, summary_text


def build_sigma_coefficient_df(
    cleaned_results: dict[float, FitResult],
    allowed_grains: list[float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    allowed = None if allowed_grains is None else {float(g) for g in allowed_grains}
    for grain, result in cleaned_results.items():
        grain_float = float(grain)
        if allowed is not None and grain_float not in allowed:
            continue
        grain_size = GRAIN_SIZE_MM.get(grain_float)
        if grain_size is None:
            continue
        params = result.params.set_index("Параметр модели")["Значение"].to_dict()
        rows.append(
            {
                "G": grain_float,
                "grain_size_mm": grain_size,
                "ln_grain_size": float(np.log(grain_size)),
                "log_a": float(params.get("log_a", np.nan)),
                "p_exp": float(params.get("p_exp", np.nan)),
                "m_exp": float(params.get("m_exp", np.nan)),
                "R²": float(result.metrics.get("R²", np.nan)),
                "RMSE_sigma": float(sigma_metric_summary(result.data).get("RMSE по cσ, %", np.nan)),
            }
        )
    return pd.DataFrame(rows).dropna().sort_values(by=["G"]).reset_index(drop=True)


def predict_temperature_diameter_universal(params: dict[str, float], D: float, tau: float, grain_size_mm: float) -> float:
    ln_g = np.log(grain_size_mm)
    a_val = params["alpha0"] + params["alpha1"] * ln_g + params["alpha2"] * (ln_g ** 2)
    if is_finite_number(params.get("b_const", np.nan)):
        b_val = params["b_const"]
    else:
        b_val = params["beta0"] + params["beta1"] * ln_g + params["beta2"] * (ln_g ** 2)
    c_val = params["gamma0"] + params["gamma1"] * ln_g + params["gamma2"] * (ln_g ** 2)
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
    log_a = params["alpha0"] + params["alpha1"] * ln_g + params["alpha2"] * (ln_g ** 2)
    p_exp = params["p_const"]
    m_exp = params["m_const"]
    if not np.isfinite(m_exp) or abs(m_exp) < 1e-12:
        raise ValueError("Универсальная sigma-модель дала слишком малый показатель степени m.")
    denom = np.exp(log_a) * np.power(max(tau, 1e-12), p_exp)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Универсальная sigma-модель дала некорректный множитель A·τ^p.")
    temp_norm = np.power(max(c_sigma / denom, 1e-12), 1.0 / m_exp)
    return float(np.clip(550.0 + 350.0 * temp_norm, 550.0, 900.0))


def evaluate_sigma_universal_model(params: dict[str, float], cleaned_results: dict[float, FitResult]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    total_points = 0
    total_models = 0
    for grain, result in cleaned_results.items():
        if float(grain) not in SIGMA_UNIVERSAL_GRAINS:
            continue
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        total_models += 1
        total_points += int(result.metrics.get("Количество точек", len(result.data)))
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
        "Количество точек": float(total_points),
        "Количество зерновых моделей": float(total_models),
        "R² по T": float(r2_score(eval_df["T"], eval_df["T_pred_universal"])) if len(eval_df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(eval_df["T"], eval_df["T_pred_universal"]))),
        "MAE по T, °C": float(mean_absolute_error(eval_df["T"], eval_df["T_pred_universal"])),
        "MAPE по T, %": float(np.mean(np.abs(errors) / np.maximum(np.abs(eval_df["T"]), 1e-9)) * 100.0),
    }


def evaluate_diameter_universal_model(params: dict[str, float], cleaned_results: dict[float, FitResult]) -> dict[str, float]:
    rows: list[pd.DataFrame] = []
    total_points = 0
    total_models = 0
    for grain, result in cleaned_results.items():
        grain_size = GRAIN_SIZE_MM.get(float(grain))
        if grain_size is None:
            continue
        total_models += 1
        total_points += int(result.metrics.get("Количество точек", len(result.data)))
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
        "Количество точек": float(total_points),
        "Количество зерновых моделей": float(total_models),
        "R² по T": float(r2_score(eval_df["T"], eval_df["T_pred_universal"])) if len(eval_df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(eval_df["T"], eval_df["T_pred_universal"]))),
        "MAE по T, °C": float(mean_absolute_error(eval_df["T"], eval_df["T_pred_universal"])),
        "MAPE по T, %": float(np.mean(np.abs(errors) / np.maximum(np.abs(eval_df["T"]), 1e-9)) * 100.0),
    }


def predict_temperature_sigma_formula(grain_number: float, c_sigma: float, tau: float) -> float:
    if c_sigma <= 0:
        raise ValueError("Для второй модели по проценту содержание sigma-фазы должно быть больше нуля.")
    if tau <= 0:
        raise ValueError("Для второй модели по проценту время должно быть больше нуля.")
    if float(grain_number) not in GRAIN_SIZE_MM:
        raise ValueError("Sigma-формула поддерживает только номера зерна 3, 4, 5, 6, 7, 8, 9 и 10.")
    g26 = -4.0 * float(grain_number) ** 2 - 36.848 * float(grain_number) + 1941.6
    return float(g26 * np.power(float(c_sigma) / np.sqrt(float(tau)), 0.192))


def build_sigma_formula_evaluation(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    eval_df = prepared_df[prepared_df["G"].isin(sorted(GRAIN_SIZE_MM.keys()))].copy()
    selected = st.session_state.get("applied_exclude_sigma_formula", [])
    if selected:
        eval_df = eval_df[~eval_df["point_id"].astype(str).isin(selected)].copy()
    if eval_df.empty:
        raise ValueError("Нет точек с поддерживаемыми номерами зерна для второй модели по проценту.")

    eval_df["T_pred_universal"] = eval_df.apply(
        lambda row: predict_temperature_sigma_formula(float(row["G"]), float(row["c_sigma"]), float(row["tau"])),
        axis=1,
    )
    eval_df["error_celsius"] = eval_df["T"] - eval_df["T_pred_universal"]
    eval_df["abs_error"] = np.abs(eval_df["error_celsius"])
    eval_df["rel_error_pct"] = np.where(
        eval_df["T"] != 0,
        eval_df["abs_error"] / np.maximum(np.abs(eval_df["T"]), 1e-9) * 100.0,
        np.nan,
    )
    err_std = float(eval_df["error_celsius"].std(ddof=0)) if len(eval_df) > 1 else 0.0
    if err_std > 0:
        eval_df["standard_residual"] = (eval_df["error_celsius"] - float(eval_df["error_celsius"].mean())) / err_std
    else:
        eval_df["standard_residual"] = 0.0

    recommendation_df = eval_df[
        (eval_df["abs_error"] >= eval_df["abs_error"].quantile(0.9))
        | (np.abs(eval_df["standard_residual"]) > 2)
        | (eval_df["rel_error_pct"] >= eval_df["rel_error_pct"].quantile(0.9))
    ].copy().sort_values(by=["abs_error", "rel_error_pct"], ascending=[False, False])

    metrics = {
        "Количество точек": float(len(eval_df)),
        "Количество зерновых моделей": float(eval_df["G"].nunique()),
        "R² по T": float(r2_score(eval_df["T"], eval_df["T_pred_universal"])) if len(eval_df) >= 2 else np.nan,
        "RMSE по T, °C": float(np.sqrt(mean_squared_error(eval_df["T"], eval_df["T_pred_universal"]))),
        "MAE по T, °C": float(mean_absolute_error(eval_df["T"], eval_df["T_pred_universal"])),
        "MAPE по T, %": float(np.mean(np.abs(eval_df["error_celsius"]) / np.maximum(np.abs(eval_df["T"]), 1e-9)) * 100.0),
    }
    return eval_df, metrics, recommendation_df


def evaluate_sigma_formula_model(prepared_df: pd.DataFrame) -> dict[str, float]:
    _, metrics, _ = build_sigma_formula_evaluation(prepared_df)
    return metrics


def parse_optional_float(value: str) -> float | None:
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def format_temperature_interpretation(temp_value: float, min_valid_temp: float = 550.0) -> str:
    if not np.isfinite(temp_value):
        return "—"
    if temp_value < min_valid_temp:
        return f"< {min_valid_temp:.0f} °C (вне физически обоснованной области)"
    return f"{temp_value:.4f}"


def is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def fmt_trimmed(value: object, decimals: int) -> str:
    if not is_finite_number(value):
        return "—"
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def add_temperature_interpretation_column(
    df: pd.DataFrame,
    source_col: str = "T_pred",
    target_col: str = "Интерпретация T_pred, °C",
    min_valid_temp: float = 550.0,
) -> pd.DataFrame:
    view = df.copy()
    if source_col in view.columns:
        view[target_col] = view[source_col].apply(lambda x: format_temperature_interpretation(float(x), min_valid_temp))
    return view


def dataframe_to_tsv_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "[пусто]"
    buffer = StringIO()
    df.to_csv(buffer, sep="\t", index=False)
    return buffer.getvalue().strip()


def round_if_present(df: pd.DataFrame, columns: list[str], decimals: int) -> pd.DataFrame:
    view = df.copy()
    for col in columns:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(decimals)
    return view


def format_columns_as_strings(df: pd.DataFrame, columns: list[str], decimals: int) -> pd.DataFrame:
    view = df.copy()
    for col in columns:
        if col in view.columns:
            numeric = pd.to_numeric(view[col], errors="coerce")
            view[col] = numeric.apply(lambda x: "" if pd.isna(x) else fmt_trimmed(float(x), decimals))
    return view


def make_export_friendly_tables(
    dataset_summary_df: pd.DataFrame,
    dataset_by_grain_df: pd.DataFrame,
    sigma_grain_df: pd.DataFrame,
    sigma_universal_df: pd.DataFrame,
    diameter_grain_df: pd.DataFrame,
    diameter_universal_df: pd.DataFrame,
    sigma_rounding_df: pd.DataFrame,
    diameter_rounding_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_fmt = format_columns_as_strings(dataset_summary_df, ["T_min, °C", "T_max, °C", "cσ_min, %", "cσ_max, %", "D_min", "D_max"], 2)
    by_grain_fmt = format_columns_as_strings(dataset_by_grain_df, ["T_min, °C", "T_max, °C", "cσ_min, %", "cσ_max, %", "D_min", "D_max"], 2)

    sigma_grain_fmt = format_columns_as_strings(
        sigma_grain_df,
        ["Размер зерна, мм", "log(A)", "p", "m", "R² по T", "R² по cσ", "RMSE по cσ, %", "MAE по cσ, %", "MAPE по cσ, %"],
        2,
    )
    sigma_grain_fmt = format_columns_as_strings(sigma_grain_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    sigma_universal_fmt = format_columns_as_strings(
        sigma_universal_df,
        ["alpha0", "alpha1", "alpha2", "p", "m", "R² для log(A)(dg)", "R² по T"],
        2,
    )
    sigma_universal_fmt = format_columns_as_strings(sigma_universal_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    diameter_grain_fmt = format_columns_as_strings(
        diameter_grain_df,
        ["Размер зерна, мм", "a", "b", "c", "R² по T"],
        2,
    )
    diameter_grain_fmt = format_columns_as_strings(diameter_grain_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    diameter_universal_fmt = format_columns_as_strings(
        diameter_universal_df,
        ["alpha0", "alpha1", "alpha2", "beta0", "beta1", "beta2", "gamma0", "gamma1", "gamma2"],
        4,
    )
    diameter_universal_fmt = format_columns_as_strings(diameter_universal_fmt, ["R² для a(dg)", "R² для b(dg)", "R² для c(dg)", "R² по T"], 2)
    diameter_universal_fmt = format_columns_as_strings(diameter_universal_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    sigma_rounding_fmt = format_columns_as_strings(sigma_rounding_df, ["R² по T", "ΔR²"], 2)
    sigma_rounding_fmt = format_columns_as_strings(sigma_rounding_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %", "ΔRMSE, °C", "ΔMAE, °C", "ΔMAPE, %"], 0)

    diameter_rounding_fmt = format_columns_as_strings(diameter_rounding_df, ["R² по T", "ΔR²"], 2)
    diameter_rounding_fmt = format_columns_as_strings(diameter_rounding_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %", "ΔRMSE, °C", "ΔMAE, °C", "ΔMAPE, %"], 0)

    return summary_fmt, by_grain_fmt, sigma_grain_fmt, sigma_universal_fmt, diameter_grain_fmt, diameter_universal_fmt, sigma_rounding_fmt, diameter_rounding_fmt


def format_report_tables(
    sigma_grain_df: pd.DataFrame,
    sigma_universal_df: pd.DataFrame,
    diameter_grain_df: pd.DataFrame,
    diameter_universal_df: pd.DataFrame,
    sigma_rounding_df: pd.DataFrame,
    diameter_rounding_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sigma_grain_fmt = sigma_grain_df.copy()
    sigma_grain_fmt = round_if_present(sigma_grain_fmt, ["Размер зерна, мм", "log(A)", "p", "m", "R² по T", "R² по cσ", "RMSE по cσ, %", "MAE по cσ, %", "MAPE по cσ, %"], 2)
    sigma_grain_fmt = round_if_present(sigma_grain_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    sigma_universal_fmt = sigma_universal_df.copy()
    sigma_universal_fmt = round_if_present(sigma_universal_fmt, ["alpha0", "alpha1", "alpha2", "p", "m", "R² для log(A)(dg)", "R² по T"], 2)
    sigma_universal_fmt = round_if_present(sigma_universal_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    diameter_grain_fmt = diameter_grain_df.copy()
    diameter_grain_fmt = round_if_present(diameter_grain_fmt, ["Размер зерна, мм", "a", "b", "c", "R² по T"], 2)
    diameter_grain_fmt = round_if_present(diameter_grain_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    diameter_universal_fmt = diameter_universal_df.copy()
    diameter_universal_fmt = round_if_present(diameter_universal_fmt, ["alpha0", "alpha1", "alpha2", "beta0", "beta1", "beta2", "gamma0", "gamma1", "gamma2"], 4)
    diameter_universal_fmt = round_if_present(diameter_universal_fmt, ["R² для a(dg)", "R² для b(dg)", "R² для c(dg)", "R² по T"], 2)
    diameter_universal_fmt = round_if_present(diameter_universal_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %"], 0)

    sigma_rounding_fmt = sigma_rounding_df.copy()
    sigma_rounding_fmt = round_if_present(sigma_rounding_fmt, ["R² по T", "ΔR²"], 2)
    sigma_rounding_fmt = round_if_present(sigma_rounding_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %", "ΔRMSE, °C", "ΔMAE, °C", "ΔMAPE, %"], 0)

    diameter_rounding_fmt = diameter_rounding_df.copy()
    diameter_rounding_fmt = round_if_present(diameter_rounding_fmt, ["R² по T", "ΔR²"], 2)
    diameter_rounding_fmt = round_if_present(diameter_rounding_fmt, ["RMSE по T, °C", "MAE по T, °C", "MAPE по T, %", "ΔRMSE, °C", "ΔMAE, °C", "ΔMAPE, %"], 0)

    return sigma_grain_fmt, sigma_universal_fmt, diameter_grain_fmt, diameter_universal_fmt, sigma_rounding_fmt, diameter_rounding_fmt


def build_dataset_summary(prepared_df: pd.DataFrame, valid_grains: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame(
        [
            {
                "Всего точек": int(len(prepared_df)),
                "Число номеров зерна": int(prepared_df["G"].nunique()),
                "Номера зерна для моделирования": ", ".join(str(int(g)) if float(g).is_integer() else str(g) for g in valid_grains),
                "T_min, °C": float(prepared_df["T"].min()),
                "T_max, °C": float(prepared_df["T"].max()),
                "tau_min": float(prepared_df["tau"].min()),
                "tau_max": float(prepared_df["tau"].max()),
                "cσ_min, %": float(prepared_df["c_sigma"].min()),
                "cσ_max, %": float(prepared_df["c_sigma"].max()),
                "D_min": float(prepared_df["D"].min()),
                "D_max": float(prepared_df["D"].max()),
            }
        ]
    )
    per_grain_df = (
        prepared_df.groupby("G", dropna=True)
        .agg(
            **{
                "Количество точек": ("G", "size"),
                "T_min, °C": ("T", "min"),
                "T_max, °C": ("T", "max"),
                "tau_min": ("tau", "min"),
                "tau_max": ("tau", "max"),
                "cσ_min, %": ("c_sigma", "min"),
                "cσ_max, %": ("c_sigma", "max"),
                "D_min": ("D", "min"),
                "D_max": ("D", "max"),
            }
        )
        .reset_index()
        .rename(columns={"G": "Номер зерна"})
        .sort_values(by=["Номер зерна"])
        .reset_index(drop=True)
    )
    return summary_df, per_grain_df


def build_grain_size_mapping_df(valid_grains: list[float] | None = None) -> pd.DataFrame:
    rows = []
    grains = sorted(valid_grains) if valid_grains else sorted(GRAIN_SIZE_MM.keys())
    for grain in grains:
        rows.append(
            {
                "Номер зерна G": float(grain),
                "Размер зерна d_g, мм": float(GRAIN_SIZE_MM.get(float(grain), np.nan)),
            }
        )
    return pd.DataFrame(rows)


def grain_mapping_caption(valid_grains: list[float] | None = None) -> str:
    grains = sorted(valid_grains) if valid_grains else sorted(GRAIN_SIZE_MM.keys())
    parts = []
    for grain in grains:
        dg = GRAIN_SIZE_MM.get(float(grain), np.nan)
        if is_finite_number(dg):
            parts.append(f"G={int(grain) if float(grain).is_integer() else grain} → d_g={fmt_trimmed(dg, 3)} мм")
    return "; ".join(parts)


def build_sigma_grain_report(cleaned_sigma_results: dict[float, FitResult]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for grain in sorted(cleaned_sigma_results.keys()):
        result = cleaned_sigma_results[grain]
        params = result.params.set_index("Параметр модели")["Значение"].to_dict()
        temp_metrics = temperature_metric_summary(result.data)
        sigma_metrics = sigma_metric_summary(result.data)
        rows.append(
            {
                "Номер зерна": float(grain),
                "Размер зерна, мм": float(GRAIN_SIZE_MM.get(float(grain), np.nan)),
                "Количество точек": float(result.metrics.get("Количество точек", len(result.data))),
                "log(A)": float(params.get("log_a", np.nan)),
                "p": float(params.get("p_exp", np.nan)),
                "m": float(params.get("m_exp", np.nan)),
                "R² по T": float(temp_metrics.get("R² по T", np.nan)),
                "RMSE по T, °C": float(temp_metrics.get("RMSE по T, °C", np.nan)),
                "MAE по T, °C": float(temp_metrics.get("MAE по T, °C", np.nan)),
                "MAPE по T, %": float(temp_metrics.get("MAPE по T, %", np.nan)),
                "R² по cσ": float(sigma_metrics.get("R² по cσ", np.nan)),
                "RMSE по cσ, %": float(sigma_metrics.get("RMSE по cσ, %", np.nan)),
                "MAE по cσ, %": float(sigma_metrics.get("MAE по cσ, %", np.nan)),
                "MAPE по cσ, %": float(sigma_metrics.get("MAPE по cσ, %", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def build_diameter_grain_report(cleaned_diameter_results: dict[float, FitResult]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for grain in sorted(cleaned_diameter_results.keys()):
        result = cleaned_diameter_results[grain]
        params = result.params.set_index("Параметр модели")["Значение"].to_dict()
        rows.append(
            {
                "Номер зерна": float(grain),
                "Размер зерна, мм": float(GRAIN_SIZE_MM.get(float(grain), np.nan)),
                "Количество точек": float(result.metrics.get("Количество точек", len(result.data))),
                "a": float(params.get("const", np.nan)),
                "b": float(params.get("ln_tau", np.nan)),
                "c": float(params.get("inv_T", np.nan)),
                "R² по T": float(result.metrics.get("R²", np.nan)),
                "RMSE по T, °C": float(result.metrics.get("RMSE, °C", np.nan)),
                "MAE по T, °C": float(result.metrics.get("MAE, °C", np.nan)),
                "MAPE по T, %": float(result.metrics.get("MAPE, %", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def build_report_export_text(
    dataset_summary_df: pd.DataFrame,
    dataset_by_grain_df: pd.DataFrame,
    sigma_grain_df: pd.DataFrame,
    sigma_universal_df: pd.DataFrame,
    diameter_grain_df: pd.DataFrame,
    diameter_universal_df: pd.DataFrame,
    sigma_rounding_df: pd.DataFrame,
    diameter_rounding_df: pd.DataFrame,
) -> str:
    sections = [
        "[ОБЩАЯ СВОДКА ПО ВЫБОРКЕ]",
        dataframe_to_tsv_text(dataset_summary_df),
        "",
        "[ВЫБОРКА ПО НОМЕРАМ ЗЕРНА]",
        dataframe_to_tsv_text(dataset_by_grain_df),
        "",
        "[ЛОКАЛЬНЫЕ SIGMA-МОДЕЛИ ПО ЗЕРНАМ]",
        dataframe_to_tsv_text(sigma_grain_df),
        "",
        "[УНИВЕРСАЛЬНАЯ SIGMA-МОДЕЛЬ]",
        dataframe_to_tsv_text(sigma_universal_df),
        "",
        "[ЛОКАЛЬНЫЕ МОДЕЛИ РОСТА ДИАМЕТРА ПО ЗЕРНАМ]",
        dataframe_to_tsv_text(diameter_grain_df),
        "",
        "[УНИВЕРСАЛЬНАЯ МОДЕЛЬ РОСТА ДИАМЕТРА]",
        dataframe_to_tsv_text(diameter_universal_df),
        "",
        "[АНАЛИЗ ОКРУГЛЕНИЯ КОЭФФИЦИЕНТОВ SIGMA-МОДЕЛИ]",
        dataframe_to_tsv_text(sigma_rounding_df),
        "",
        "[АНАЛИЗ ОКРУГЛЕНИЯ КОЭФФИЦИЕНТОВ МОДЕЛИ РОСТА ДИАМЕТРА]",
        dataframe_to_tsv_text(diameter_rounding_df),
    ]
    return "\n".join(sections)


def rounded_copy(params: dict[str, float], decimals: int, keys: list[str]) -> dict[str, float]:
    rounded = dict(params)
    for key in keys:
        if key in rounded and is_finite_number(rounded[key]):
            rounded[key] = round(float(rounded[key]), decimals)
    return rounded


def get_final_sigma_params(params: dict[str, float]) -> dict[str, float]:
    return rounded_copy(params, 2, ["alpha0", "alpha1", "alpha2", "p_const", "m_const"])


def get_final_diameter_params(params: dict[str, float]) -> dict[str, float]:
    return rounded_copy(params, 4, ["alpha0", "alpha1", "alpha2", "beta0", "beta1", "beta2", "gamma0", "gamma1", "gamma2"])


def build_sigma_rounding_analysis(params: dict[str, float], cleaned_sigma_results: dict[float, FitResult]) -> pd.DataFrame:
    base_eval = evaluate_sigma_universal_model(params, cleaned_sigma_results)
    rows: list[dict[str, float]] = [
        {
            "Знаков после запятой": 10,
            "R² по T": float(base_eval["R² по T"]),
            "RMSE по T, °C": float(base_eval["RMSE по T, °C"]),
            "MAE по T, °C": float(base_eval["MAE по T, °C"]),
            "MAPE по T, %": float(base_eval["MAPE по T, %"]),
            "ΔR²": 0.0,
            "ΔRMSE, °C": 0.0,
            "ΔMAE, °C": 0.0,
            "ΔMAPE, %": 0.0,
        }
    ]
    sigma_keys = ["alpha0", "alpha1", "alpha2", "p_const", "m_const"]
    for decimals in [8, 7, 6, 5, 4, 3, 2, 1]:
        rounded_params = rounded_copy(params, decimals, sigma_keys)
        eval_item = evaluate_sigma_universal_model(rounded_params, cleaned_sigma_results)
        rows.append(
            {
                "Знаков после запятой": decimals,
                "R² по T": float(eval_item["R² по T"]),
                "RMSE по T, °C": float(eval_item["RMSE по T, °C"]),
                "MAE по T, °C": float(eval_item["MAE по T, °C"]),
                "MAPE по T, %": float(eval_item["MAPE по T, %"]),
                "ΔR²": float(eval_item["R² по T"] - base_eval["R² по T"]),
                "ΔRMSE, °C": float(eval_item["RMSE по T, °C"] - base_eval["RMSE по T, °C"]),
                "ΔMAE, °C": float(eval_item["MAE по T, °C"] - base_eval["MAE по T, °C"]),
                "ΔMAPE, %": float(eval_item["MAPE по T, %"] - base_eval["MAPE по T, %"]),
            }
        )
    return pd.DataFrame(rows)


def build_diameter_rounding_analysis(params: dict[str, float], cleaned_diameter_results: dict[float, FitResult]) -> pd.DataFrame:
    base_eval = evaluate_diameter_universal_model(params, cleaned_diameter_results)
    rows: list[dict[str, float]] = [
        {
            "Знаков после запятой": 10,
            "R² по T": float(base_eval["R² по T"]),
            "RMSE по T, °C": float(base_eval["RMSE по T, °C"]),
            "MAE по T, °C": float(base_eval["MAE по T, °C"]),
            "MAPE по T, %": float(base_eval["MAPE по T, %"]),
            "ΔR²": 0.0,
            "ΔRMSE, °C": 0.0,
            "ΔMAE, °C": 0.0,
            "ΔMAPE, %": 0.0,
        }
    ]
    diameter_keys = ["alpha0", "alpha1", "alpha2", "beta0", "beta1", "beta2", "gamma0", "gamma1", "gamma2"]
    for decimals in [8, 7, 6, 5, 4, 3, 2, 1]:
        rounded_params = rounded_copy(params, decimals, diameter_keys)
        eval_item = evaluate_diameter_universal_model(rounded_params, cleaned_diameter_results)
        rows.append(
            {
                "Знаков после запятой": decimals,
                "R² по T": float(eval_item["R² по T"]),
                "RMSE по T, °C": float(eval_item["RMSE по T, °C"]),
                "MAE по T, °C": float(eval_item["MAE по T, °C"]),
                "MAPE по T, %": float(eval_item["MAPE по T, %"]),
                "ΔR²": float(eval_item["R² по T"] - base_eval["R² по T"]),
                "ΔRMSE, °C": float(eval_item["RMSE по T, °C"] - base_eval["RMSE по T, °C"]),
                "ΔMAE, °C": float(eval_item["MAE по T, °C"] - base_eval["MAE по T, °C"]),
                "ΔMAPE, %": float(eval_item["MAPE по T, %"] - base_eval["MAPE по T, %"]),
            }
        )
    return pd.DataFrame(rows)


def render_report_data_tab(prepared_df: pd.DataFrame, valid_grains: list[float]) -> None:
    st.subheader("Данные для научного отчета / диссертации")
    st.caption("Вкладка собирает ключевые таблицы по выборке, локальным и универсальным моделям в удобном для копирования виде.")

    if not valid_grains:
        st.warning("Для подготовки отчета недостаточно зерновых наборов с минимум 7 точками.")
        return

    dataset_summary_df, dataset_by_grain_df = build_dataset_summary(prepared_df, valid_grains)
    grain_size_mapping_df = build_grain_size_mapping_df(valid_grains)
    cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
    cleaned_diameter_results = build_cleaned_diameter_grain_results(prepared_df, valid_grains)

    sigma_grain_df = build_sigma_grain_report(cleaned_sigma_results)
    diameter_grain_df = build_diameter_grain_report(cleaned_diameter_results)

    sigma_params_raw, _, _ = fit_sigma_universal_grain_size_model(cleaned_sigma_results, variant="median_constants")
    sigma_params = get_final_sigma_params(sigma_params_raw)
    sigma_eval = evaluate_sigma_universal_model(sigma_params, cleaned_sigma_results)
    sigma_universal_df = pd.DataFrame(
        [
            {
                "Формула": "cσ = A(dg) · τ^p · ((T - 550) / 350)^m",
                "alpha0": float(sigma_params["alpha0"]),
                "alpha1": float(sigma_params["alpha1"]),
                "alpha2": float(sigma_params["alpha2"]),
                "p": float(sigma_params["p_const"]),
                "m": float(sigma_params["m_const"]),
                "R² для log(A)(dg)": float(sigma_params["r2_log_a"]),
                "R² по T": float(sigma_eval["R² по T"]),
                "RMSE по T, °C": float(sigma_eval["RMSE по T, °C"]),
                "MAE по T, °C": float(sigma_eval["MAE по T, °C"]),
                "MAPE по T, %": float(sigma_eval["MAPE по T, %"]),
                "Количество зерновых моделей": float(sigma_eval["Количество зерновых моделей"]),
                "Количество точек": float(sigma_eval["Количество точек"]),
            }
        ]
    )

    diameter_params_raw, _, _ = fit_diameter_universal_grain_size_model(cleaned_diameter_results, variant="quadratic_full")
    diameter_params = get_final_diameter_params(diameter_params_raw)
    diameter_eval = evaluate_diameter_universal_model(diameter_params, cleaned_diameter_results)
    diameter_universal_df = pd.DataFrame(
        [
            {
                "Формула": "ln(D) = a(dg) + b(dg)·ln(τ) + c(dg)·(1/T(K))",
                "alpha0": float(diameter_params["alpha0"]),
                "alpha1": float(diameter_params["alpha1"]),
                "alpha2": float(diameter_params["alpha2"]),
                "beta0": float(diameter_params["beta0"]),
                "beta1": float(diameter_params["beta1"]),
                "beta2": float(diameter_params["beta2"]),
                "gamma0": float(diameter_params["gamma0"]),
                "gamma1": float(diameter_params["gamma1"]),
                "gamma2": float(diameter_params["gamma2"]),
                "R² для a(dg)": float(diameter_params["r2_a"]),
                "R² для b(dg)": float(diameter_params["r2_b"]),
                "R² для c(dg)": float(diameter_params["r2_c"]),
                "R² по T": float(diameter_eval["R² по T"]),
                "RMSE по T, °C": float(diameter_eval["RMSE по T, °C"]),
                "MAE по T, °C": float(diameter_eval["MAE по T, °C"]),
                "MAPE по T, %": float(diameter_eval["MAPE по T, %"]),
                "Количество зерновых моделей": float(diameter_eval["Количество зерновых моделей"]),
                "Количество точек": float(diameter_eval["Количество точек"]),
            }
        ]
    )

    sigma_rounding_df = build_sigma_rounding_analysis(sigma_params_raw, cleaned_sigma_results)
    diameter_rounding_df = build_diameter_rounding_analysis(diameter_params_raw, cleaned_diameter_results)
    (
        sigma_grain_df_fmt,
        sigma_universal_df_fmt,
        diameter_grain_df_fmt,
        diameter_universal_df_fmt,
        sigma_rounding_df_fmt,
        diameter_rounding_df_fmt,
    ) = format_report_tables(
        sigma_grain_df,
        sigma_universal_df,
        diameter_grain_df,
        diameter_universal_df,
        sigma_rounding_df,
        diameter_rounding_df,
    )
    (
        dataset_summary_export,
        dataset_by_grain_export,
        sigma_grain_export,
        sigma_universal_export,
        diameter_grain_export,
        diameter_universal_export,
        sigma_rounding_export,
        diameter_rounding_export,
    ) = make_export_friendly_tables(
        dataset_summary_df,
        dataset_by_grain_df,
        sigma_grain_df_fmt,
        sigma_universal_df_fmt,
        diameter_grain_df_fmt,
        diameter_universal_df_fmt,
        sigma_rounding_df_fmt,
        diameter_rounding_df_fmt,
    )

    st.markdown("**1. Общая сводка по выборке**")
    st.dataframe(dataset_summary_df, use_container_width=True, hide_index=True)

    st.markdown("**2. Выборка по номерам зерна**")
    st.dataframe(dataset_by_grain_df, use_container_width=True, hide_index=True)

    st.markdown("**2а. Соответствие номера зерна G и размера зерна d_g**")
    st.dataframe(grain_size_mapping_df, use_container_width=True, hide_index=True)
    st.caption(grain_mapping_caption(valid_grains))

    st.markdown("**3. Локальные sigma-модели по зернам**")
    st.dataframe(sigma_grain_df_fmt, use_container_width=True, hide_index=True)

    st.markdown("**4. Универсальная sigma-модель**")
    st.dataframe(sigma_universal_df_fmt, use_container_width=True, hide_index=True)

    st.markdown("**5. Локальные модели роста диаметра по зернам**")
    st.dataframe(diameter_grain_df_fmt, use_container_width=True, hide_index=True)

    st.markdown("**6. Универсальная модель роста диаметра**")
    st.dataframe(diameter_universal_df_fmt, use_container_width=True, hide_index=True)

    st.markdown("**7. Анализ округления коэффициентов универсальной sigma-модели**")
    st.caption("Сравнение качества модели при последовательном сокращении числа знаков после запятой у итоговых коэффициентов.")
    st.dataframe(sigma_rounding_df_fmt, use_container_width=True, hide_index=True)

    st.markdown("**8. Анализ округления коэффициентов универсальной модели роста диаметра**")
    st.caption("Таблица помогает выбрать минимальную длину записи коэффициентов без заметной потери точности.")
    st.dataframe(diameter_rounding_df_fmt, use_container_width=True, hide_index=True)

    export_text = build_report_export_text(
        dataset_summary_export,
        dataset_by_grain_export,
        sigma_grain_export,
        sigma_universal_export,
        diameter_grain_export,
        diameter_universal_export,
        sigma_rounding_export,
        diameter_rounding_export,
    )
    with st.expander("Текстовый экспорт для ассистента"):
        st.caption("Этот блок можно целиком скопировать и прислать в чат для подготовки научного отчета.")
        st.text_area("Скопируйте данные ниже", export_text, height=500, key="report_export_text")


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
            st.metric("Расчетная температура по модели диаметра, °C", format_temperature_interpretation(temp_value))
            if temp_value < 550.0:
                st.caption("Значение ниже 550 °C показывается как вне физически обоснованной области, но сама формула модели не меняется.")
        except Exception as exc:
            st.error(f"Не удалось выполнить расчет по модели диаметра: {exc}")

    show_result_block(
        result,
        key_prefix=f"diameter_grain_{grain_value}",
        include_grain=False,
        fit_function=fit_diameter_growth_model,
        preselect_outliers=True,
        auto_apply_selected=False,
        min_interpretable_temp=550.0,
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


def show_model_comparison(
    base_result: FitResult,
    improved_result: FitResult,
    anchor_result: FitResult,
    diameter_result: FitResult | None = None,
) -> None:
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
    comparison_payload = {
        "Метрика": metric_order,
        "Базовая модель": [base_result.metrics.get(metric, np.nan) for metric in metric_order],
        "Улучшенная модель": [improved_result.metrics.get(metric, np.nan) for metric in metric_order],
        "Sigma-модель по зернам": [anchor_result.metrics.get(metric, np.nan) for metric in metric_order],
    }
    if diameter_result is not None:
        comparison_payload["Модель роста диаметра"] = [diameter_result.metrics.get(metric, np.nan) for metric in metric_order]
    comparison_df = pd.DataFrame(comparison_payload)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    anchor_compare_rows = [
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
    if diameter_result is not None:
        anchor_compare_rows.append(
            {
                "Модель": "Модель роста диаметра",
                "Прогноз для реальной точки, °C": diameter_result.metrics.get("Прогноз для реальной точки, °C", np.nan),
                "Отклонение от диапазона 570–600 °C, °C": diameter_result.metrics.get(
                    "Отклонение реальной точки от диапазона, °C", np.nan
                ),
            }
        )
    anchor_compare = pd.DataFrame(anchor_compare_rows)
    st.subheader("Проверка по важной реальной точке")
    st.dataframe(anchor_compare, use_container_width=True, hide_index=True)


def show_multi_calculator(
    base_result: FitResult,
    improved_result: FitResult,
    anchor_result: FitResult,
    diameter_result: FitResult | None = None,
) -> None:
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
    diameter_params = None
    if diameter_result is not None:
        diameter_params = diameter_result.params.set_index("Параметр модели")["Значение"].to_dict()

    try:
        base_temp = predict_temperature_engineering(base_params, d_value, tau_value, sigma_value, grain_value)
        improved_temp = predict_temperature_improved(improved_params, d_value, tau_value, sigma_value, grain_value)
        anchor_temp = predict_temperature_anchor_saturation(anchor_params, d_value, tau_value, sigma_value, grain_value)

        calc_rows = [
            {"Модель": "Базовая", "Расчетная температура, °C": base_temp},
            {"Модель": "Улучшенная", "Расчетная температура, °C": improved_temp},
            {"Модель": "Sigma-модель по зернам", "Расчетная температура, °C": anchor_temp},
        ]
        if diameter_params is not None:
            diameter_temp = predict_temperature_diameter_grain_model(diameter_params, d_value, tau_value, grain_value)
            calc_rows.append({"Модель": "Модель роста диаметра", "Расчетная температура, °C": diameter_temp})

        metric_columns = st.columns(len(calc_rows))
        for col, row in zip(metric_columns, calc_rows):
            with col:
                st.metric(f"Температура: {row['Модель']}, °C", f"{row['Расчетная температура, °C']:.4f}")

        calc_df = pd.DataFrame(calc_rows)
        st.dataframe(calc_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Не удалось выполнить расчет: {exc}")


def render_calibration_tab(prepared_df: pd.DataFrame) -> None:
    st.subheader("Калибровка программы")
    st.caption(
        "Загрузите Excel/CSV с контрольными точками. Раздел сравнит предполагаемую температуру с расчетом по моделям "
        "и покажет, какая из них ближе к ожидаемым значениям."
    )
    st.info(
        "Ожидаемые столбцы: время (tau), диаметр (D), номер зерна (G), процент sigma-фазы (c_sigma), предполагаемая температура. "
        "Названия могут быть русскими или латиницей."
    )

    template_bytes = build_calibration_template_workbook()
    c_template, c_hint = st.columns([1, 2])
    with c_template:
        st.download_button(
            "Скачать шаблон Excel",
            data=template_bytes,
            file_name="shablon_kalibrovki_regressiya.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_calibration_template",
            use_container_width=True,
        )
    with c_hint:
        st.caption("В шаблоне уже есть пример строк и отдельный лист с описанием столбцов.")

    calibration_file = st.file_uploader(
        "Файл для калибровки",
        type=["xls", "xlsx", "csv"],
        key="calibration_file_uploader",
        help="Например: время, диаметр сигмы, номер зерна, процент сигмы, предполагаемая температура.",
    )

    if calibration_file is None:
        st.info("Загрузите файл калибровки, и программа покажет отклонения по всем моделям.")
        return

    try:
        calibration_raw_df = load_file(calibration_file)
        calibration_df = prepare_calibration_dataframe(calibration_raw_df)
        base_calibration_result = fit_engineering_model(prepared_df, include_grain=True)
        improved_calibration_result = fit_improved_model(prepared_df, include_grain=True)
        anchor_calibration_result = fit_anchor_saturation_model(prepared_df, include_grain=True)
        diameter_calibration_result = fit_diameter_growth_model(prepared_df, include_grain=True)
    except Exception as exc:
        st.error(f"Не удалось выполнить калибровку: {exc}")
        return

    with st.expander("Предпросмотр исходного файла калибровки"):
        st.dataframe(calibration_raw_df, use_container_width=True)

    base_params = base_calibration_result.params.set_index("Параметр модели")["Значение"].to_dict()
    improved_params = improved_calibration_result.params.set_index("Параметр модели")["Значение"].to_dict()
    anchor_params = anchor_calibration_result.params.set_index("Параметр модели")["Значение"].to_dict()
    diameter_params = diameter_calibration_result.params.set_index("Параметр модели")["Значение"].to_dict()

    def safe_apply(frame: pd.DataFrame, predictor, *columns: str) -> pd.Series:
        values: list[float] = []
        for _, row in frame.iterrows():
            try:
                args = [float(row[col]) for col in columns]
                values.append(float(predictor(*args)))
            except Exception:
                values.append(np.nan)
        return pd.Series(values, index=frame.index, dtype=float)

    result_df = calibration_df.copy()
    result_df["T_базовая, °C"] = safe_apply(
        result_df,
        lambda D, tau, G, c_sigma: predict_temperature_engineering(base_params, D, tau, c_sigma, G),
        "D",
        "tau",
        "G",
        "c_sigma",
    )
    result_df["Δ базовая, °C"] = result_df["T_базовая, °C"] - result_df["T_assumed"]
    result_df["|Δ| базовая, °C"] = result_df["Δ базовая, °C"].abs()

    result_df["T_улучшенная, °C"] = safe_apply(
        result_df,
        lambda D, tau, G, c_sigma: predict_temperature_improved(improved_params, D, tau, c_sigma, G),
        "D",
        "tau",
        "G",
        "c_sigma",
    )
    result_df["Δ улучшенная, °C"] = result_df["T_улучшенная, °C"] - result_df["T_assumed"]
    result_df["|Δ| улучшенная, °C"] = result_df["Δ улучшенная, °C"].abs()

    result_df["T_sigma по зерну, °C"] = safe_apply(
        result_df,
        lambda D, tau, G, c_sigma: predict_temperature_anchor_saturation(anchor_params, D, tau, c_sigma, G),
        "D",
        "tau",
        "G",
        "c_sigma",
    )
    result_df["Δ sigma по зерну, °C"] = result_df["T_sigma по зерну, °C"] - result_df["T_assumed"]
    result_df["|Δ| sigma по зерну, °C"] = result_df["Δ sigma по зерну, °C"].abs()

    result_df["T_рост диаметра, °C"] = safe_apply(
        result_df,
        lambda D, tau, G: predict_temperature_diameter_grain_model(diameter_params, D, tau, G),
        "D",
        "tau",
        "G",
    )
    result_df["Δ рост диаметра, °C"] = result_df["T_рост диаметра, °C"] - result_df["T_assumed"]
    result_df["|Δ| рост диаметра, °C"] = result_df["Δ рост диаметра, °C"].abs()

    abs_error_columns = {
        "Базовая модель": "|Δ| базовая, °C",
        "Улучшенная модель": "|Δ| улучшенная, °C",
        "Sigma-модель по зерну": "|Δ| sigma по зерну, °C",
        "Модель роста диаметра": "|Δ| рост диаметра, °C",
    }
    delta_columns = {
        "Базовая модель": "Δ базовая, °C",
        "Улучшенная модель": "Δ улучшенная, °C",
        "Sigma-модель по зерну": "Δ sigma по зерну, °C",
        "Модель роста диаметра": "Δ рост диаметра, °C",
    }

    def choose_best_model(row: pd.Series) -> str:
        candidates = {label: row[col] for label, col in abs_error_columns.items() if is_finite_number(row[col])}
        if not candidates:
            return "Нет расчета"
        return min(candidates, key=candidates.get)

    result_df["Лучшая модель по точке"] = result_df.apply(choose_best_model, axis=1)

    unavailable_counts = {
        "Базовая модель": int(result_df["T_базовая, °C"].isna().sum()),
        "Улучшенная модель": int(result_df["T_улучшенная, °C"].isna().sum()),
        "Sigma-модель по зерну": int(result_df["T_sigma по зерну, °C"].isna().sum()),
        "Модель роста диаметра": int(result_df["T_рост диаметра, °C"].isna().sum()),
    }
    unavailable_total = sum(unavailable_counts.values())
    if unavailable_total > 0:
        st.warning(
            "Для части строк некоторые модели не дали расчет. "
            + "; ".join(f"{label}: {count}" for label, count in unavailable_counts.items() if count > 0)
        )

    def highlight_calibration_row(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        temp_columns = [
            "T_базовая, °C",
            "T_улучшенная, °C",
            "T_sigma по зерну, °C",
            "T_рост диаметра, °C",
        ]
        delta_columns_local = [
            "Δ базовая, °C",
            "Δ улучшенная, °C",
            "Δ sigma по зерну, °C",
            "Δ рост диаметра, °C",
        ]
        abs_columns_local = [
            "|Δ| базовая, °C",
            "|Δ| улучшенная, °C",
            "|Δ| sigma по зерну, °C",
            "|Δ| рост диаметра, °C",
        ]

        finite_errors = {col: row[col] for col in abs_columns_local if is_finite_number(row[col])}
        best_error = min(finite_errors.values()) if finite_errors else None
        worst_error = max(finite_errors.values()) if finite_errors else None

        for idx, col in enumerate(row.index):
            if col in temp_columns and row.get(col) == row.get(col):
                styles[idx] = "background-color: #f6f8fa;"
            if col in delta_columns_local:
                value = row[col]
                if is_finite_number(value):
                    if abs(value) <= 10:
                        styles[idx] = "background-color: #e8f5e9; color: #1b5e20; font-weight: 600;"
                    elif abs(value) >= 30:
                        styles[idx] = "background-color: #ffebee; color: #b71c1c; font-weight: 600;"
            if col in abs_columns_local and best_error is not None and worst_error is not None:
                value = row[col]
                if not is_finite_number(value):
                    styles[idx] = "background-color: #eeeeee; color: #757575;"
                elif value == best_error:
                    styles[idx] = "background-color: #c8e6c9; color: #1b5e20; font-weight: 700;"
                elif value == worst_error:
                    styles[idx] = "background-color: #ffcdd2; color: #b71c1c; font-weight: 700;"
            if col == "Лучшая модель по точке":
                styles[idx] = "background-color: #fff3cd; color: #7a5d00; font-weight: 700;"
        return styles

    st.markdown("**Таблица калибровки по точкам**")
    calibration_view_columns = [
        "point_id",
        "tau",
        "D",
        "G",
        "c_sigma",
        "T_assumed",
        "T_базовая, °C",
        "Δ базовая, °C",
        "T_улучшенная, °C",
        "Δ улучшенная, °C",
        "T_sigma по зерну, °C",
        "Δ sigma по зерну, °C",
        "T_рост диаметра, °C",
        "Δ рост диаметра, °C",
        "Лучшая модель по точке",
    ]
    display_df = result_df[calibration_view_columns].rename(
        columns={
            "point_id": "Точка",
            "tau": "Время, ч",
            "D": "Диаметр sigma",
            "G": "Номер зерна",
            "c_sigma": "Sigma-фаза, %",
            "T_assumed": "Предполагаемая температура, °C",
        }
    )
    st.dataframe(
        display_df.style.apply(highlight_calibration_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Зелёным подсвечены лучшие/близкие значения, красным — наибольшие отклонения по строке.")

    best_model_counts = result_df["Лучшая модель по точке"].value_counts()
    metric_cols = st.columns(4)
    for col, label in zip(metric_cols, abs_error_columns.keys()):
        with col:
            hits = int(best_model_counts.get(label, 0))
            mean_abs = float(result_df[abs_error_columns[label]].mean())
            st.metric(label, f"{mean_abs:.2f} °C", f"лучших точек: {hits}")

    summary_rows = []
    for label, abs_col in abs_error_columns.items():
        signed_col = delta_columns[label]
        summary_rows.append(
            {
                "Модель": label,
                "Количество точек": len(result_df),
                "Среднее абсолютное отклонение, °C": float(result_df[abs_col].mean()),
                "Максимальное абсолютное отклонение, °C": float(result_df[abs_col].max()),
                "Среднее отклонение, °C": float(result_df[signed_col].mean()),
                "Медиана |Δ|, °C": float(result_df[abs_col].median()),
                "Лучших попаданий": int((result_df["Лучшая модель по точке"] == label).sum()),
                "Без расчета": int(result_df[abs_col].isna().sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Среднее абсолютное отклонение, °C", "Медиана |Δ|, °C", "Максимальное абсолютное отклонение, °C"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary_df.index = summary_df.index + 1

    st.markdown("**Итог по близости к предполагаемой температуре**")
    def highlight_summary_row(row: pd.Series) -> list[str]:
        if row.name == 1:
            return ["background-color: #c8e6c9; color: #1b5e20; font-weight: 700;"] * len(row)
        return [""] * len(row)

    st.dataframe(summary_df.style.apply(highlight_summary_row, axis=1), use_container_width=True)

    best_model = summary_df.iloc[0]
    st.success(
        f"Сейчас ближе всего к предполагаемым температурам работает: {best_model['Модель']}. "
        f"Среднее абсолютное отклонение = {best_model['Среднее абсолютное отклонение, °C']:.2f} °C."
    )


def render_universal_models_tab(prepared_df: pd.DataFrame, valid_grains: list[float]) -> None:
    st.subheader("Универсальные модели по размеру зерна")
    st.markdown(f"**Научное обоснование:** {SCIENTIFIC_UNIVERSAL_SIGMA_PARAGRAPH}")

    if not valid_grains:
        st.warning("Для построения универсальных моделей недостаточно зерновых наборов с минимум 7 точками.")
        return

    st.markdown("**Соответствие номера зерна и физического размера зерна**")
    st.caption("В универсальных моделях в коэффициенты подставляется не номер зерна G напрямую, а соответствующий ему физический размер зерна d_g.")
    st.dataframe(build_grain_size_mapping_df(valid_grains), use_container_width=True, hide_index=True)
    st.info(grain_mapping_caption(valid_grains))

    recommended_diameter_exclusions = get_recommended_diameter_exclusions(prepared_df, valid_grains)
    recommended_sigma_exclusions = get_recommended_sigma_exclusions(prepared_df, valid_grains)
    sigma_formula_recommended_labels: list[str] = []
    try:
        _, _, sigma_formula_recommendation_df = build_sigma_formula_evaluation(prepared_df)
        sigma_formula_recommended_labels = sigma_formula_recommendation_df["point_id"].astype(str).tolist()
    except Exception:
        sigma_formula_recommended_labels = []

    st.caption(
        "Если нажать кнопку ниже, программа автоматически применит все рекомендованные исключения по локальным моделям зерна "
        "и пересчитает универсальные модели. Если кнопку не нажимать, можно удалять точки вручную в разделах отдельных номеров зерна."
    )
    c_apply_all, c_reset_all = st.columns(2)
    with c_apply_all:
        if st.button("Удалить все предложенные точки", key="apply_all_universal_model_exclusions"):
            for grain in valid_grains:
                st.session_state[f"applied_exclude_diameter_grain_{grain}"] = list(recommended_diameter_exclusions.get(grain, []))
                st.session_state[f"applied_exclude_sigma_grain_{grain}"] = list(recommended_sigma_exclusions.get(grain, []))
            st.session_state["applied_exclude_sigma_formula"] = list(sigma_formula_recommended_labels)
            st.rerun()
    with c_reset_all:
        if st.button("Сбросить все автоисключения", key="reset_all_universal_model_exclusions"):
            for grain in valid_grains:
                st.session_state[f"applied_exclude_diameter_grain_{grain}"] = []
                st.session_state[f"applied_exclude_sigma_grain_{grain}"] = []
            st.session_state["applied_exclude_sigma_formula"] = []
            st.rerun()

    exclusion_rows = []
    for grain in valid_grains:
        exclusion_rows.append(
            {
                "Номер зерна": grain,
                "Предложено удалить (диаметр)": len(recommended_diameter_exclusions.get(grain, [])),
                "Сейчас исключено (диаметр)": len(st.session_state.get(f"applied_exclude_diameter_grain_{grain}", [])),
                "Предложено удалить (sigma)": len(recommended_sigma_exclusions.get(grain, [])),
                "Сейчас исключено (sigma)": len(st.session_state.get(f"applied_exclude_sigma_grain_{grain}", [])),
            }
        )
    st.dataframe(pd.DataFrame(exclusion_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Для второй модели по проценту рекомендовано исключить: {len(sigma_formula_recommended_labels)}; "
        f"сейчас исключено: {len(st.session_state.get('applied_exclude_sigma_formula', []))}."
    )

    diameter_error_local = None
    sigma_error_local = None
    diameter_payload = None
    selected_sigma_variant = None

    try:
        cleaned_diameter_results = build_cleaned_diameter_grain_results(prepared_df, valid_grains)
        universal_diameter_params, diameter_coeff_df, diameter_summary = fit_diameter_universal_grain_size_model(
            cleaned_diameter_results, variant="quadratic_full"
        )
        final_diameter_params = get_final_diameter_params(universal_diameter_params)
        diameter_eval = evaluate_diameter_universal_model(final_diameter_params, cleaned_diameter_results)
        diameter_payload = {
            "key": "quadratic_full",
            "title": "Диаметр: a(dg), b(dg), c(dg)",
            "params": final_diameter_params,
            "params_raw": universal_diameter_params,
            "coeff_df": diameter_coeff_df,
            "summary": diameter_summary,
            "eval": diameter_eval,
        }
    except Exception as exc:
        diameter_error_local = str(exc)

    try:
        cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
        params_item, coeff_df_item, summary_item = fit_sigma_universal_grain_size_model(cleaned_sigma_results, variant="median_constants")
        final_sigma_params = get_final_sigma_params(params_item)
        eval_item = evaluate_sigma_universal_model(final_sigma_params, cleaned_sigma_results)
        selected_sigma_variant = {
            "key": "median_constants",
            "title": "Sigma: p,m = медианы",
            "params": final_sigma_params,
            "params_raw": params_item,
            "coeff_df": coeff_df_item,
            "summary": summary_item,
            "eval": eval_item,
        }
    except Exception as exc:
        sigma_error_local = str(exc)

    sigma_formula_eval = None
    sigma_formula_error = None
    try:
        sigma_formula_eval = evaluate_sigma_formula_model(prepared_df)
    except Exception as exc:
        sigma_formula_error = str(exc)

    quality_rows = []
    if diameter_payload is not None:
        quality_rows.append(
            {
                "Модель": "Универсальная модель диаметра",
                "Версия": diameter_payload["title"],
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
    if sigma_formula_eval is not None:
        quality_rows.append(
            {
                "Модель": "Вторая модель по проценту",
                "Версия": "T = G26 · (cσ / √τ)^0.192",
                "R² по T": sigma_formula_eval["R² по T"],
                "RMSE по T, °C": sigma_formula_eval["RMSE по T, °C"],
                "MAE по T, °C": sigma_formula_eval["MAE по T, °C"],
                "MAPE по T, %": sigma_formula_eval["MAPE по T, %"],
                "Количество точек": sigma_formula_eval["Количество точек"],
            }
        )
    if quality_rows:
        st.subheader("Сравнение качества универсальных моделей")
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

    col_left, col_mid, col_right = st.columns(3)
    with col_left:
        st.subheader("Формула универсальной модели диаметра")
        if diameter_payload is None:
            st.error(f"Модель диаметра недоступна: {diameter_error_local}")
        else:
            p = diameter_payload["params"]
            st.code(
                "\n".join(
                    [
                        "ln(D) = a(dg) + b(dg)·ln(τ) + c(dg)·(1/T(K))",
                        f"a(dg) = {fmt_trimmed(p['alpha0'], 4)} + ({fmt_trimmed(p['alpha1'], 4)}) · ln(dg) + ({fmt_trimmed(p['alpha2'], 4)}) · [ln(dg)]²",
                        f"b(dg) = {fmt_trimmed(p['beta0'], 4)} + ({fmt_trimmed(p['beta1'], 4)}) · ln(dg) + ({fmt_trimmed(p['beta2'], 4)}) · [ln(dg)]²",
                        f"c(dg) = {fmt_trimmed(p['gamma0'], 4)} + ({fmt_trimmed(p['gamma1'], 4)}) · ln(dg) + ({fmt_trimmed(p['gamma2'], 4)}) · [ln(dg)]²",
                    ]
                ),
                language="text",
            )
            with st.expander("Сводка по универсальной модели диаметра"):
                st.text(diameter_payload["summary"])

    with col_mid:
        st.subheader("Формула универсальной sigma-модели")
        if selected_sigma_variant is None:
            st.error(f"Sigma-модель недоступна: {sigma_error_local}")
        else:
            p = selected_sigma_variant["params"]
            st.code(
                "\n".join(
                    [
                        "cσ = A(dg) · τ^p · ((T - 550) / 350)^m",
                        f"log(A)(dg) = {fmt_trimmed(p['alpha0'], 2)} + ({fmt_trimmed(p['alpha1'], 2)}) · ln(dg) + ({fmt_trimmed(p['alpha2'], 2)}) · [ln(dg)]²",
                        f"p = {fmt_trimmed(p['p_const'], 2)}",
                        f"m = {fmt_trimmed(p['m_const'], 2)}",
                    ]
                ),
                language="text",
            )
            with st.expander("Сводка по выбранной универсальной sigma-модели"):
                st.text(selected_sigma_variant["summary"])

    with col_right:
        st.subheader("Формула второй модели по проценту")
        if sigma_formula_eval is None:
            st.error(f"Вторая модель по проценту недоступна: {sigma_formula_error}")
        else:
            st.code(
                "\n".join(
                    [
                        "T = G26 · (cσ / √τ)^0.192",
                        "G26 = -4·G² - 36.848·G + 1941.6",
                        "G = 3, 4, 5, 6, 7, 8, 9, 10",
                    ]
                ),
                language="text",
            )
            st.caption(
                f"Оценка на всех доступных загруженных точках: R²={sigma_formula_eval['R² по T']:.4f}, "
                f"RMSE={sigma_formula_eval['RMSE по T, °C']:.4f} °C, точек={int(sigma_formula_eval['Количество точек'])}."
            )

    st.subheader("Исключение точек для второй модели по проценту")
    if sigma_formula_eval is None:
        st.error(f"Не удалось оценить вторую модель по проценту: {sigma_formula_error}")
    else:
        sigma_formula_options = prepared_df[prepared_df["G"].isin(sorted(GRAIN_SIZE_MM.keys()))]["point_id"].astype(str).tolist()
        selected_sigma_formula = st.multiselect(
            "Исключить точки из второй модели по проценту",
            options=sigma_formula_options,
            default=st.session_state.get("applied_exclude_sigma_formula", sigma_formula_recommended_labels),
            key="exclude_sigma_formula",
        )
        c_apply_formula, c_reset_formula = st.columns(2)
        with c_apply_formula:
            if st.button("Применить исключение выбранных точек для второй модели по проценту", key="apply_sigma_formula_exclusions"):
                st.session_state["applied_exclude_sigma_formula"] = list(selected_sigma_formula)
                st.rerun()
        with c_reset_formula:
            if st.button("Сбросить исключения второй модели по проценту", key="reset_sigma_formula_exclusions"):
                st.session_state["applied_exclude_sigma_formula"] = []
                st.rerun()

    st.subheader("Калькулятор по универсальным моделям")
    st.caption("Можно ввести диаметр, процент σ-фазы или оба параметра сразу. Расчет запускается только по кнопке.")
    st.caption(f"Соответствие для расчета: {grain_mapping_caption(valid_grains)}")
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
                result_rows.append(
                    {
                        "Модель": "Универсальная модель диаметра",
                        "Расчетная температура, °C": temp_d,
                        "Интерпретация": format_temperature_interpretation(temp_d),
                    }
                )
            if sigma_value is not None:
                if selected_sigma_variant is None:
                    raise ValueError(sigma_error_local or "Универсальная sigma-модель недоступна.")
                temp_sigma = predict_temperature_sigma_universal(selected_sigma_variant["params"], tau_value, round(sigma_value, 2), grain_size)
                result_rows.append(
                    {
                        "Модель": "Универсальная sigma-модель",
                        "Расчетная температура, °C": temp_sigma,
                        "Интерпретация": f"{temp_sigma:.4f}",
                    }
                )
                temp_sigma_formula = predict_temperature_sigma_formula(float(grain_number), round(sigma_value, 2), tau_value)
                result_rows.append(
                    {
                        "Модель": "Вторая модель по проценту",
                        "Расчетная температура, °C": temp_sigma_formula,
                        "Интерпретация": f"{temp_sigma_formula:.4f}",
                    }
                )

            if len(result_rows) == 1:
                row = result_rows[0]
                st.success(f"{row['Модель']}: {row['Интерпретация']}")
            else:
                result_columns = st.columns(len(result_rows))
                for col, row in zip(result_columns, result_rows):
                    with col:
                        st.metric(f"{row['Модель']}, °C", row["Интерпретация"])
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Не удалось выполнить расчет: {exc}")


def render_sigma_formula_tab(prepared_df: pd.DataFrame) -> None:
    st.subheader("Вторая модель по проценту")
    st.write(
        "Отдельный раздел для второй модели по проценту: здесь собраны формула, качество, "
        "таблица по всем точкам и рекомендации по исключению выбросов."
    )

    try:
        sigma_formula_df, sigma_formula_metrics, sigma_formula_recommendation_df = build_sigma_formula_evaluation(prepared_df)
    except Exception as exc:
        st.error(f"Не удалось оценить вторую модель по проценту: {exc}")
        return

    st.code(
        "\n".join(
            [
                "T = G26 · (cσ / √τ)^0.192",
                "G26 = -4·G² - 36.848·G + 1941.6",
                "G = 3, 4, 5, 6, 7, 8, 9, 10",
            ]
        ),
        language="text",
    )

    metric_df = pd.DataFrame([sigma_formula_metrics])
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    if not sigma_formula_recommendation_df.empty:
        st.warning("Ниже точки, которые система рекомендует проверить или временно исключить из второй модели по проценту.")
    sigma_formula_options = prepared_df[prepared_df["G"].isin(sorted(GRAIN_SIZE_MM.keys()))]["point_id"].astype(str).tolist()
    selected_sigma_formula = st.multiselect(
        "Исключить точки из второй модели по проценту",
        options=sigma_formula_options,
        default=st.session_state.get("applied_exclude_sigma_formula", sigma_formula_recommendation_df["point_id"].astype(str).tolist()),
        key="sigma_formula_tab_exclude",
    )
    c_apply_formula, c_reset_formula = st.columns(2)
    with c_apply_formula:
        if st.button("Применить исключение выбранных точек", key="sigma_formula_tab_apply"):
            st.session_state["applied_exclude_sigma_formula"] = list(selected_sigma_formula)
            st.rerun()
    with c_reset_formula:
        if st.button("Сбросить исключения", key="sigma_formula_tab_reset"):
            st.session_state["applied_exclude_sigma_formula"] = []
            st.rerun()

    st.subheader("Таблица по всем точкам")
    st.dataframe(
        sigma_formula_df[
            ["point_id", "G", "tau", "c_sigma", "T", "T_pred_universal", "error_celsius", "abs_error", "rel_error_pct", "standard_residual"]
        ].rename(
            columns={
                "point_id": "Точка",
                "G": "Номер зерна",
                "tau": "τ",
                "c_sigma": "cσ, %",
                "T": "T факт, °C",
                "T_pred_universal": "T расчёт, °C",
                "error_celsius": "Ошибка, °C",
                "abs_error": "|Ошибка|, °C",
                "rel_error_pct": "Ошибка, %",
                "standard_residual": "Станд. остаток",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not sigma_formula_recommendation_df.empty:
        st.subheader("Рекомендованные к проверке/удалению точки")
        st.dataframe(
            sigma_formula_recommendation_df[
                ["point_id", "G", "T", "T_pred_universal", "error_celsius", "abs_error", "rel_error_pct", "standard_residual"]
            ].rename(
                columns={
                    "point_id": "Точка",
                    "G": "Номер зерна",
                    "T": "T факт, °C",
                    "T_pred_universal": "T расчёт, °C",
                    "error_celsius": "Ошибка, °C",
                    "abs_error": "|Ошибка|, °C",
                    "rel_error_pct": "Ошибка, %",
                    "standard_residual": "Станд. остаток",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_result_block(
    result: FitResult,
    key_prefix: str = "main",
    include_grain: bool = True,
    fit_function=fit_engineering_model,
    preselect_outliers: bool = True,
    auto_apply_selected: bool = True,
    min_interpretable_temp: float | None = None,
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
    ].head(15).copy()
    if min_interpretable_temp is not None:
        weak_view = add_temperature_interpretation_column(weak_view, "T_pred", min_valid_temp=min_interpretable_temp)
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
    result_view = result.data[view_columns].copy()
    if min_interpretable_temp is not None:
        result_view = add_temperature_interpretation_column(result_view, "T_pred", min_valid_temp=min_interpretable_temp)
    st.dataframe(result_view, use_container_width=True, hide_index=True)

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
    enrich_real_point_metrics(diameter_result, lambda params, D, tau, c_sigma, G: predict_temperature_diameter_grain_model(params, D, tau, G))
if anchor_result is not None:
    enrich_real_point_metrics(anchor_result, predict_temperature_anchor_saturation)

main_tab, grain_tab, improved_tab, diameter_tab, anchor_tab, compare_tab, calculator_tab, calibration_tab, universal_models_tab, sigma_formula_tab, report_tab = st.tabs([
    "Общая модель",
    "Модели по номерам зерна",
    "Улучшенная модель",
    "Рост диаметра",
    "Простая sigma-модель",
    "Сравнение моделей",
    "Калькулятор",
    "Калибровка программы",
    "Универсальные модели",
    "Вторая модель по проценту",
    "Данные для отчета",
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
                universal_params, coeff_df, universal_summary = fit_diameter_universal_grain_size_model(
                    cleaned_diameter_results, variant="quadratic_full"
                )
                universal_eval = evaluate_diameter_universal_model(universal_params, cleaned_diameter_results)
                meta_quality_df = pd.DataFrame(
                    [
                        {
                            "R² для a(dg)": universal_params["r2_a"],
                            "R² для b(dg)": universal_params["r2_b"],
                            "R² для c(dg)": universal_params["r2_c"],
                            "R² по T": universal_eval["R² по T"],
                            "RMSE по T, °C": universal_eval["RMSE по T, °C"],
                            "Количество точек": universal_eval["Количество точек"],
                        }
                    ]
                )
                st.dataframe(meta_quality_df, use_container_width=True, hide_index=True)
                formula_text = (
                    "ln(D) = a(dg) + b(dg)·ln(τ) + c(dg)·(1/T(K))\n"
                    "a(dg) = alpha0 + alpha1·ln(dg) + alpha2·[ln(dg)]²\n"
                    "b(dg) = beta0 + beta1·ln(dg) + beta2·[ln(dg)]²\n"
                    "c(dg) = gamma0 + gamma1·ln(dg) + gamma2·[ln(dg)]²\n"
                    f"alpha0 = {fmt_trimmed(universal_params['alpha0'], 4)}, alpha1 = {fmt_trimmed(universal_params['alpha1'], 4)}, alpha2 = {fmt_trimmed(universal_params['alpha2'], 4)}\n"
                    f"beta0 = {fmt_trimmed(universal_params['beta0'], 4)}, beta1 = {fmt_trimmed(universal_params['beta1'], 4)}, beta2 = {fmt_trimmed(universal_params['beta2'], 4)}\n"
                    f"gamma0 = {fmt_trimmed(universal_params['gamma0'], 4)}, gamma1 = {fmt_trimmed(universal_params['gamma1'], 4)}, gamma2 = {fmt_trimmed(universal_params['gamma2'], 4)}"
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
                        st.metric("Расчетная температура по универсальной модели, °C", format_temperature_interpretation(temp_value))
                        if temp_value < 550.0:
                            st.caption("Если расчет ниже 550 °C, результат показывается как вне физически обоснованной области без изменения самой формулы модели.")
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
        local_tab, universal_tab = st.tabs(["Модели по отдельным зернам", "Универсальная модель"])
        with local_tab:
            cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
            selected_sigma_grain = st.selectbox("Выберите номер зерна для sigma-модели", valid_grains)
            for grain in valid_grains:
                grain_df = prepared_df[prepared_df["G"] == grain].copy()
                try:
                    grain_result = cleaned_sigma_results.get(grain) or fit_anchor_saturation_model(grain_df, include_grain=False)
                    sigma_metrics = sigma_metric_summary(grain_result.data)
                    sigma_grain_scores.append(
                        {
                            "Номер зерна": grain,
                            "Количество точек": grain_result.metrics["Количество точек"],
                            **sigma_metrics,
                        }
                    )
                    if grain == selected_sigma_grain:
                        show_sigma_grain_block(grain_result, grain, grain_df)
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

        with universal_tab:
            recommended_exclusions = get_recommended_sigma_exclusions(prepared_df, valid_grains)
            selected_grains_for_recommended = st.multiselect(
                "Для каких номеров зерна добавить рекомендованные системой точки в исключения",
                options=valid_grains,
                default=[],
                key="sigma_recommended_grains_picker",
            )
            c_apply_all, c_reset_all = st.columns(2)
            with c_apply_all:
                if st.button("Применить рекомендованные точки для выбранных зерен", key="apply_selected_sigma_exclusions"):
                    for grain in selected_grains_for_recommended:
                        labels = recommended_exclusions.get(grain, [])
                        apply_key = f"applied_exclude_sigma_grain_{grain}"
                        merged = sorted(set(st.session_state.get(apply_key, [])) | set(labels))
                        st.session_state[apply_key] = merged
                        st.session_state[f"pending_sync_sigma_grain_{grain}"] = True
                    st.rerun()
            with c_reset_all:
                if st.button("Сбросить все исключения по sigma-модели", key="reset_all_sigma_exclusions"):
                    for grain in valid_grains:
                        st.session_state[f"applied_exclude_sigma_grain_{grain}"] = []
                        st.session_state[f"pending_sync_sigma_grain_{grain}"] = True
                    st.rerun()

            active_rows = []
            for grain in valid_grains:
                active_rows.append(
                    {
                        "Номер зерна": grain,
                        "Рекомендовано исключить": len(recommended_exclusions.get(grain, [])),
                        "Сейчас исключено": len(st.session_state.get(f"applied_exclude_sigma_grain_{grain}", [])),
                    }
                )
            st.dataframe(pd.DataFrame(active_rows), use_container_width=True, hide_index=True)

            st.subheader("Универсальная sigma-модель по размеру зерна")
            st.write(
                "Подход повторяет универсальную модель роста диаметра: сначала строятся отдельные sigma-модели "
                "для каждого номера зерна, затем для log(A) берется фиксированная форма "
                "u0 + u1·ln(dg) + u2·[ln(dg)]², а p и m заменяются на константы. "
                "В общую sigma-модель включены все пять доступных зерен: 3, 5, 8, 9 и 10."
            )
            try:
                cleaned_sigma_results = build_cleaned_sigma_grain_results(prepared_df, valid_grains)
                sigma_coeff_all = build_sigma_coefficient_df(cleaned_sigma_results, allowed_grains=valid_grains)
                if not sigma_coeff_all.empty:
                    st.markdown("**Сначала — обзор коэффициентов по всем доступным номерам зерна, уже по очищенным локальным sigma-моделям.**")
                    coeff_all_view = sigma_coeff_all[["G", "grain_size_mm", "log_a", "p_exp", "m_exp", "R²", "RMSE_sigma"]].rename(
                        columns={
                            "grain_size_mm": "Размер зерна, мм",
                            "log_a": "log(A)",
                            "p_exp": "p",
                            "m_exp": "m",
                            "R²": "R² по T",
                            "RMSE_sigma": "RMSE по cσ, %",
                        }
                    )
                    st.dataframe(coeff_all_view, use_container_width=True, hide_index=True)
                selected_params, sigma_coeff_df, sigma_summary = fit_sigma_universal_grain_size_model(cleaned_sigma_results, variant="median_constants")
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

                meta_quality_df = pd.DataFrame(
                    [
                        {
                            "R² для log(A)(dg)": selected_params["r2_log_a"],
                            "p": selected_params["p_const"],
                            "m": selected_params["m_const"],
                            "R² по T": sigma_eval["R² по T"],
                            "RMSE по T, °C": sigma_eval["RMSE по T, °C"],
                            "Количество зерновых моделей": sigma_eval["Количество зерновых моделей"],
                            "Количество точек": sigma_eval["Количество точек"],
                        }
                    ]
                )
                st.dataframe(meta_quality_df, use_container_width=True, hide_index=True)

                st.info(
                    f"Для общей sigma-модели используются все пять зерен: 3, 5, 8, 9, 10, причем коэффициенты берутся из очищенных локальных sigma-моделей. "
                    f"Форма для log(A) зафиксирована: u0 + u1·ln(dg) + u2·[ln(dg)]². "
                    f"Для универсальной модели с медианными p и m: RMSE={sigma_eval['RMSE по T, °C']:.4f} °C, "
                    f"R²={sigma_eval['R² по T']:.4f}."
                )

                st.code(
                    "\n".join(
                        [
                            f"log(A)(dg) = {fmt_trimmed(selected_params['alpha0'], 2)} + ({fmt_trimmed(selected_params['alpha1'], 2)}) · ln(dg) + ({fmt_trimmed(selected_params['alpha2'], 2)}) · [ln(dg)]²",
                            f"p = {fmt_trimmed(selected_params['p_const'], 2)}",
                            f"m = {fmt_trimmed(selected_params['m_const'], 2)}",
                            "",
                            "cσ = A(dg) · τ^p · ((T - 550) / 350)^m",
                        ]
                    ),
                    language="text",
                )
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
        show_model_comparison(base_result, improved_result, anchor_result, diameter_result)

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
        show_multi_calculator(base_result, improved_result, anchor_result, diameter_result)

with calibration_tab:
    render_calibration_tab(prepared_df)

with universal_models_tab:
    render_universal_models_tab(prepared_df, valid_grains)

with sigma_formula_tab:
    render_sigma_formula_tab(prepared_df)

with report_tab:
    render_report_data_tab(prepared_df, valid_grains)
