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


def fit_anchor_saturation_model(df: pd.DataFrame, include_grain: bool = True) -> FitResult:
    if len(df) < 7:
        raise ValueError("Для устойчивой подгонки нужно хотя бы 7 точек.")

    tau = df["tau"].to_numpy(dtype=float)
    inv_t = df["inv_T"].to_numpy(dtype=float)
    grain = df["G"].to_numpy(dtype=float) if include_grain else None
    x_true = df["c_sigma"].to_numpy(dtype=float)

    def jmak_sigma(params: np.ndarray, tau_vals: np.ndarray, inv_t_vals: np.ndarray, grain_vals: np.ndarray | None) -> np.ndarray:
        log_k0, q_like, p_exp, n_exp = params[:4]
        grain_coeff = params[4] if include_grain else 0.0
        log_rate = log_k0 + q_like * inv_t_vals + p_exp * np.log(tau_vals)
        if grain_vals is not None:
            log_rate += grain_coeff * grain_vals
        rate_term = np.exp(np.clip(log_rate, -60, 60))
        transformed = np.power(rate_term, n_exp)
        return SIGMA_SATURATION_LIMIT * (1.0 - np.exp(-transformed))

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = jmak_sigma(params, tau, inv_t, grain)
        penalty = 1e-4 * np.sum(np.square(params))
        return np.append(pred - x_true, np.sqrt(penalty))

    if include_grain:
        x0 = np.array([-12.0, 8000.0, 0.35, 1.5, 0.0], dtype=float)
        lower = np.array([-40.0, -20000.0, 0.01, 0.2, -2.0], dtype=float)
        upper = np.array([10.0, 30000.0, 3.0, 6.0, 2.0], dtype=float)
        param_rows = [("k0", "log_k0"), ("Q*", "q_like"), ("p", "p_exp"), ("n", "n_exp"), ("g", "grain_coeff")]
    else:
        x0 = np.array([-12.0, 8000.0, 0.35, 1.5], dtype=float)
        lower = np.array([-40.0, -20000.0, 0.01, 0.2], dtype=float)
        upper = np.array([10.0, 30000.0, 3.0, 6.0], dtype=float)
        param_rows = [("k0", "log_k0"), ("Q*", "q_like"), ("p", "p_exp"), ("n", "n_exp")]

    fit = optimize.least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        max_nfev=20000,
    )
    if not fit.success:
        raise ValueError(f"JMAK/Avrami подгонка не сошлась: {fit.message}")

    params_vector = fit.x
    x_pred = jmak_sigma(params_vector, tau, inv_t, grain)

    def solve_temperature_from_sigma(target_sigma: float, tau_value: float, grain_value: float | None = None) -> float:
        if target_sigma <= 0 or target_sigma >= SIGMA_SATURATION_LIMIT:
            raise ValueError("Для обратного расчета содержание сигма-фазы должно быть в диапазоне (0, 18).")

        def f(temp_c: float) -> float:
            temp_k = temp_c + 273.15
            inv_temp = 1.0 / temp_k
            grain_array = None if grain_value is None else np.array([grain_value], dtype=float)
            pred = jmak_sigma(params_vector, np.array([tau_value], dtype=float), np.array([inv_temp], dtype=float), grain_array)[0]
            return float(pred - target_sigma)

        t_min = 550.0
        t_max = 900.0
        grid = np.linspace(t_min, t_max, 200)
        values = [f(t) for t in grid]
        for left, right, v_left, v_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if v_left == 0:
                return float(left)
            if v_left * v_right < 0:
                return float(optimize.brentq(f, left, right))

        best_idx = int(np.argmin(np.abs(values)))
        return float(grid[best_idx])

    fitted_c = np.array(
        [solve_temperature_from_sigma(x, t, None if grain is None else g) for x, t, g in zip(x_true, tau, grain if grain is not None else np.zeros(len(df)))],
        dtype=float,
    )
    fitted_kelvin = fitted_c + 273.15
    fitted_inv_t = 1.0 / fitted_kelvin

    result_df = df.copy()
    result_df["inv_T_pred"] = fitted_inv_t
    result_df["T_pred"] = fitted_c
    result_df["sigma_pred_pct"] = x_pred
    result_df["error_celsius"] = result_df["T"] - result_df["T_pred"]
    result_df["abs_error"] = np.abs(result_df["error_celsius"])
    result_df["rel_error_pct"] = np.where(
        result_df["T"] != 0,
        result_df["abs_error"] / np.abs(result_df["T"]) * 100,
        np.nan,
    )
    std_err = result_df["error_celsius"].std(ddof=1)
    if np.isfinite(std_err) and std_err > 0:
        result_df["standard_residual"] = (result_df["error_celsius"] - result_df["error_celsius"].mean()) / std_err
    else:
        result_df["standard_residual"] = 0.0
    result_df["leverage"] = np.nan
    result_df["cooks_distance"] = np.nan

    weak_points = result_df.sort_values(
        by=["abs_error", "cooks_distance", "standard_residual"], ascending=[False, False, False]
    ).copy()
    outlier_recommendation = weak_points[
        (weak_points["abs_error"] >= weak_points["abs_error"].quantile(0.9))
        | (np.abs(weak_points["standard_residual"]) > 2)
    ].copy()

    param_names = [name for _, name in param_rows]
    params = pd.DataFrame(
        {
            "Коэффициент": [label for label, _ in param_rows],
            "Параметр модели": param_names,
            "Значение": list(params_vector),
            "StdErr": [np.nan] * len(param_rows),
            "t-статистика": [np.nan] * len(param_rows),
            "p-value": [np.nan] * len(param_rows),
            "Нижняя 95% граница": [np.nan] * len(param_rows),
            "Верхняя 95% граница": [np.nan] * len(param_rows),
        }
    )

    metrics = build_metrics(result_df, predictor_count=len(param_rows))
    metrics["RMSE модели сигма-фазы, %"] = float(np.sqrt(mean_squared_error(x_true, x_pred)))

    formula_text = (
        "cσ = 18·{1 - exp[-(k(T,G)·τ^p)^n]}\n"
        "ln k(T,G) = log_k0 + q_like·(1/T(K))"
        + (" + grain_coeff·G" if include_grain else "")
        + "\nТемпература восстанавливается численно из cσ в окне 550–900 °C"
    )

    summary_text = (
        "JMAK/Avrami-модель только по содержанию сигма-фазы.\n\n"
        f"Предельное содержание сигма-фазы зафиксировано на уровне {SIGMA_SATURATION_LIMIT:.2f}%. "
        "Прямая модель описывает рост фазы во времени, а температура восстанавливается обратным численным подбором в окне 550–900 °C.\n"
        f"Статус подгонки: success={fit.success}, cost={float(fit.cost):.6f}, nfev={fit.nfev}, message={fit.message}."
    )

    return FitResult(
        data=result_df,
        metrics=metrics,
        params=params,
        weak_points=weak_points,
        model_summary=summary_text,
        outlier_recommendation=outlier_recommendation,
        formula_text=formula_text,
        model_label="JMAK/Avrami модель по содержанию сигма-фазы",
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


def predict_temperature_anchor_saturation(
    params: dict[str, float], D: float, tau: float, c_sigma: float, G: float | None = None
) -> float:
    if c_sigma <= 0 or c_sigma >= SIGMA_SATURATION_LIMIT:
        raise ValueError("Для JMAK/Avrami модели содержание сигма-фазы должно быть в диапазоне (0, 18).")

    log_k0 = params.get("log_k0", np.nan)
    q_like = params.get("q_like", np.nan)
    p_exp = params.get("p_exp", np.nan)
    n_exp = params.get("n_exp", np.nan)
    grain_coeff = params.get("grain_coeff", 0.0)

    def sigma_at_temp(temp_c: float) -> float:
        inv_temp = 1.0 / (temp_c + 273.15)
        log_rate = log_k0 + q_like * inv_temp + p_exp * np.log(tau)
        if G is not None:
            log_rate += grain_coeff * G
        rate_term = np.exp(np.clip(log_rate, -60, 60))
        transformed = rate_term**n_exp
        return float(SIGMA_SATURATION_LIMIT * (1.0 - np.exp(-transformed)))

    def f(temp_c: float) -> float:
        return sigma_at_temp(temp_c) - c_sigma

    t_min, t_max = 550.0, 900.0
    grid = np.linspace(t_min, t_max, 200)
    values = [f(t) for t in grid]
    for left, right, v_left, v_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if v_left == 0:
            return float(left)
        if v_left * v_right < 0:
            return float(optimize.brentq(f, left, right))

    best_idx = int(np.argmin(np.abs(values)))
    return float(grid[best_idx])


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
            "JMAK/Avrami sigma-модель": [anchor_result.metrics.get(metric, np.nan) for metric in metric_order],
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
                "Модель": "JMAK/Avrami sigma-модель"},{
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
    st.caption("Введите параметры структуры и наработки — программа сразу посчитает температуру по базовой, улучшенной и JMAK/Avrami модели по сигма-фазе.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tau_value = st.number_input("Время наработки τ", min_value=1e-9, value=1000.0, step=100.0, format="%.6f")
    with c2:
        d_value = st.number_input("Эквивалентный диаметр D", min_value=1e-9, value=10.0, step=0.1, format="%.6f")
    with c3:
        sigma_value = st.number_input("Содержание сигма-фазы cσ, %", min_value=1e-9, value=1.0, step=0.1, format="%.6f")
    with c4:
        grain_value = st.number_input("Номер зерна G", value=8.0, step=1.0, format="%.6f")

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
            st.metric("Температура по JMAK/Avrami модели, °C", f"{anchor_temp:.4f}")

        calc_df = pd.DataFrame(
            [
                {"Модель": "Базовая", "Расчетная температура, °C": base_temp},
                {"Модель": "Улучшенная", "Расчетная температура, °C": improved_temp},
                {"Модель": "JMAK/Avrami sigma-модель", "Расчетная температура, °C": anchor_temp},
            ]
        )
        st.dataframe(calc_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Не удалось выполнить расчет: {exc}")


def show_result_block(
    result: FitResult,
    key_prefix: str = "main",
    include_grain: bool = True,
    fit_function=fit_engineering_model,
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
            default=outlier_labels,
            key=f"exclude_{key_prefix}",
        )
        if selected:
            filtered = result.data[~result.data["point_id"].astype(str).isin(selected)].copy()
            if len(filtered) >= 7:
                st.info(f"Пересчет после исключения {len(selected)} точек.")
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
if anchor_result is not None:
    enrich_real_point_metrics(anchor_result, predict_temperature_anchor_saturation)

main_tab, grain_tab, improved_tab, anchor_tab, compare_tab, calculator_tab = st.tabs([
    "Общая модель",
    "Модели по номерам зерна",
    "Улучшенная модель",
    "JMAK/Avrami sigma-модель",
    "Сравнение моделей",
    "Калькулятор",
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

with anchor_tab:
    st.write(
        "Новая модель использует только содержание сигма-фазы в форме JMAK/Avrami. "
        "Рост сигма-фазы ограничен пределом 18%, а температура восстанавливается не прямой формулой, а численным подбором в окне 550–900 °C. "
        "Это ближе к физике превращения, чем прежняя линейная обратная зависимость."
    )
    try:
        if anchor_result is None:
            raise ValueError(anchor_error or "неизвестная ошибка")
        show_result_block(
            anchor_result,
            key_prefix="anchor_all",
            include_grain=True,
            fit_function=fit_anchor_saturation_model,
        )
        st.info(
            f"Прогноз для реальной точки (τ={REAL_WORLD_POINT['tau']:.0f} ч, D={REAL_WORLD_POINT['D']}, "
            f"cσ={REAL_WORLD_POINT['c_sigma']}, G={REAL_WORLD_POINT['G']:.0f}) = "
            f"{anchor_result.metrics.get('Прогноз для реальной точки, °C', np.nan):.4f} °C."
        )
    except Exception as exc:
        st.error(f"Не удалось построить JMAK/Avrami модель: {exc}")

with compare_tab:
    if base_result is None:
        st.error(f"Базовая модель недоступна для сравнения: {base_error}")
    elif improved_result is None:
        st.error(f"Улучшенная модель недоступна для сравнения: {improved_error}")
    elif anchor_result is None:
        st.error(f"JMAK/Avrami модель недоступна для сравнения: {anchor_error}")
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
                        "R² JMAK/Avrami": anchor_grain_result.metrics["R²"],
                        "RMSE базовая, °C": base_grain_result.metrics["RMSE, °C"],
                        "RMSE улучшенная, °C": improved_grain_result.metrics["RMSE, °C"],
                        "RMSE JMAK/Avrami, °C": anchor_grain_result.metrics["RMSE, °C"],
                        "MAPE базовая, %": base_grain_result.metrics["MAPE, %"],
                        "MAPE улучшенная, %": improved_grain_result.metrics["MAPE, %"],
                        "MAPE JMAK/Avrami, %": anchor_grain_result.metrics["MAPE, %"],
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
        st.error(f"Калькулятор JMAK/Avrami модели недоступен: {anchor_error}")
    else:
        show_multi_calculator(base_result, improved_result, anchor_result)
