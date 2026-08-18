#!/usr/bin/env python3
"""Reproduce the laser-ablation depth modelling workflow.

The script is the non-interactive counterpart to
``notebooks/random_forest_analysis.ipynb``. It validates the tidy experimental
data, compares baseline regressors, tunes a random forest using training-set
cross-validation, evaluates the selected model once on a held-out test set,
and writes the tables and figures used to interpret the result.

Run from any working directory:

    python src/train_model.py

Use ``--output-root`` to keep a trial run separate from the tracked project
results, for example:

    python src/train_model.py --output-root /tmp/laser-ablation-run
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


LOGGER = logging.getLogger("laser_ablation_ml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "ablation_data.csv"
FEATURES = ["voltage_V", "pulses"]
TARGET = "depth_um"
DEFAULT_RANDOM_STATE = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the laser-ablation depth regressors."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Tidy CSV input (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=(
            "Root directory for results/ and figures/ "
            f"(default: {REPOSITORY_ROOT})."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Seed used for splitting and stochastic models (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for random forests and grid search (default: -1, all cores).",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def configure_plots() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160})


def prepare_output_directories(output_root: Path) -> tuple[Path, Path]:
    """Create and return the results and figures directories."""
    root = output_root.expanduser().resolve()
    result_dir = root / "results"
    figure_dir = root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, figure_dir


def load_and_validate_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the tidy CSV and fail early on conditions that invalidate modelling."""
    path = data_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}")

    data = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")

    missing_values = int(data[required].isna().sum().sum())
    non_finite_values = int((~np.isfinite(data[required].to_numpy())).sum())
    duplicate_settings = int(data.duplicated(FEATURES).sum())
    if missing_values:
        raise ValueError(f"Found {missing_values} missing values in modelling columns.")
    if non_finite_values:
        raise ValueError(f"Found {non_finite_values} non-finite modelling values.")
    if duplicate_settings:
        raise ValueError(f"Found {duplicate_settings} duplicate voltage-pulse settings.")
    if len(data) < 20:
        raise ValueError("At least 20 observations are required for this workflow.")

    quality = pd.DataFrame(
        [
            {
                "rows": len(data),
                "missing_values": missing_values,
                "duplicate_settings": duplicate_settings,
                "voltage_levels": data["voltage_V"].nunique(),
                "pulse_levels": data["pulses"].nunique(),
                "minimum_voltage_V": data["voltage_V"].min(),
                "maximum_voltage_V": data["voltage_V"].max(),
                "minimum_pulses": data["pulses"].min(),
                "maximum_pulses": data["pulses"].max(),
                "minimum_depth_um": data[TARGET].min(),
                "maximum_depth_um": data[TARGET].max(),
            }
        ]
    )
    return data, quality


def make_models(random_state: int, n_jobs: int) -> dict[str, RegressorMixin]:
    """Construct the baseline models used in the notebook comparison."""
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Quadratic regression": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        ),
        "Reference random forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
    }


def make_cv(random_state: int) -> KFold:
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


def compare_models(
    models: Mapping[str, RegressorMixin],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
) -> pd.DataFrame:
    """Compare candidate models using training-set cross-validation only."""
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=cv, scoring=SCORING)
        rows.append(
            {
                "model": name,
                "R2_mean": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "MAE_mean_um": -scores["test_mae"].mean(),
                "RMSE_mean_um": -scores["test_rmse"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE_mean_um").reset_index(drop=True)


def plot_cv_comparison(cv_results: pd.DataFrame, output_path: Path) -> None:
    ordered = cv_results.sort_values("RMSE_mean_um", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(ordered["model"], ordered["RMSE_mean_um"], color="#3b82b8")
    ax.set_xlabel("Mean cross-validation RMSE (μm)")
    ax.set_title("Model comparison on the training set")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
    random_state: int,
    n_jobs: int,
) -> tuple[RandomForestRegressor, pd.DataFrame, float]:
    """Tune the forest without exposing the held-out test set."""
    parameter_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt"],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    # An unrestricted forest and max_depth=15 are numerically tied on this
    # dataset. Select deterministically among effectively equal scores, with a
    # finite shallower depth preferred over an unrestricted tree. This keeps
    # the command-line result aligned with the documented notebook result even
    # when parallel execution or scikit-learn versions differ in the last bit.
    mean_scores = np.asarray(search.cv_results_["mean_test_score"])
    tied = np.flatnonzero(
        np.isclose(mean_scores, search.best_score_, rtol=0.0, atol=1e-10)
    )

    def tie_break_key(index: int) -> tuple[float, int, int]:
        params = search.cv_results_["params"][index]
        depth = params["max_depth"]
        finite_depth = float(depth) if depth is not None else float("inf")
        return (
            finite_depth,
            -int(params["min_samples_leaf"]),
            int(params["n_estimators"]),
        )

    selected_index = min(tied, key=tie_break_key)
    selected_params = search.cv_results_["params"][selected_index]
    selected_model = RandomForestRegressor(
        random_state=random_state, n_jobs=n_jobs, **selected_params
    )
    selected_model.fit(x_train, y_train)

    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
        "param_max_features",
    ]
    tuning_results = pd.DataFrame(search.cv_results_)[columns].copy()
    tuning_results["CV_RMSE_um"] = -tuning_results.pop("mean_test_score")
    tuning_results = tuning_results.sort_values("rank_test_score").reset_index(drop=True)
    best_cv_rmse = float(-mean_scores[selected_index])
    return selected_model, tuning_results, best_cv_rmse


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_um": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_um": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_models(
    models: Mapping[str, RegressorMixin],
    final_model: RandomForestRegressor,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Fit each model and report train and held-out test metrics."""
    evaluation_models = {**models, "Tuned random forest": final_model}
    rows: list[dict[str, Any]] = []
    for name, model in evaluation_models.items():
        model.fit(x_train, y_train)
        rows.append(
            {
                "model": name,
                "split": "train",
                **regression_metrics(y_train, model.predict(x_train)),
            }
        )
        rows.append(
            {
                "model": name,
                "split": "test",
                **regression_metrics(y_test, model.predict(x_test)),
            }
        )
    return pd.DataFrame(rows)


def make_test_predictions(
    final_model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame]:
    predictions = final_model.predict(x_test)
    result = x_test.copy()
    result["measured_depth_um"] = y_test
    result["predicted_depth_um"] = predictions
    result["residual_um"] = y_test - predictions
    result["absolute_error_um"] = result["residual_um"].abs()
    result = result.sort_values("absolute_error_um", ascending=False)
    return predictions, result


def plot_test_diagnostics(
    y_test: pd.Series,
    predictions: np.ndarray,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(y_test, predictions, s=45, alpha=0.8)
    limits = [
        min(float(y_test.min()), float(predictions.min())) - 0.5,
        max(float(y_test.max()), float(predictions.max())) + 0.5,
    ]
    ax.plot(limits, limits, "--", color="black", lw=1)
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Measured depth (μm)",
        ylabel="Predicted depth (μm)",
        title="Measured and predicted depth",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "predicted_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(predictions, y_test.to_numpy() - predictions, s=45, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Predicted depth (μm)")
    ax.set_ylabel("Residual: measured - predicted (μm)")
    ax.set_title("Test-set residuals")
    fig.tight_layout()
    fig.savefig(figure_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig)


def calculate_feature_importance(
    final_model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    permutation = permutation_importance(
        final_model,
        x_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="r2",
    )
    positive = np.maximum(permutation.importances_mean, 0)
    total_positive = positive.sum()
    fractions = positive / total_positive if total_positive else np.zeros_like(positive)
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "impurity_fraction": final_model.feature_importances_,
            "permutation_R2_drop": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
            "permutation_fraction": fractions,
        }
    )


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    plot_data = importance.set_index("feature")[
        ["impurity_fraction", "permutation_fraction"]
    ]
    ax = plot_data.plot(
        kind="bar", figsize=(6.5, 4.5), rot=0, color=["#3b82b8", "#e58b3a"]
    )
    ax.set_ylabel("Relative importance")
    ax.set_title("Feature importance")
    ax.legend(["Tree impurity", "Test permutation"])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def grouped_validation(
    model: RandomForestRegressor,
    x: pd.DataFrame,
    y: pd.Series,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate interpolation difficulty when complete parameter levels are unseen."""
    rows: list[dict[str, Any]] = []
    checks = [
        ("Unseen voltage levels", data["voltage_V"], 4),
        ("Unseen pulse levels", data["pulses"], 5),
    ]
    for label, groups, folds in checks:
        scores = cross_validate(
            model,
            x,
            y,
            groups=groups,
            cv=GroupKFold(n_splits=folds),
            scoring=SCORING,
        )
        rows.append(
            {
                "check": label,
                "R2_mean": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "MAE_mean_um": -scores["test_mae"].mean(),
                "RMSE_mean_um": -scores["test_rmse"].mean(),
            }
        )
    return pd.DataFrame(rows)


def physical_consistency(
    final_model: RandomForestRegressor, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check local monotonic behaviour on the measured voltage-pulse grid."""
    voltage_values = np.sort(data["voltage_V"].unique())
    pulse_values = np.sort(data["pulses"].unique())
    grid = pd.MultiIndex.from_product(
        [voltage_values, pulse_values], names=FEATURES
    ).to_frame(index=False)
    grid["predicted_depth_um"] = final_model.predict(grid[FEATURES])
    prediction_table = grid.pivot(
        index="voltage_V", columns="pulses", values="predicted_depth_um"
    )

    voltage_comparisons = (len(voltage_values) - 1) * len(pulse_values)
    pulse_comparisons = len(voltage_values) * (len(pulse_values) - 1)
    voltage_drops = int((np.diff(prediction_table.values, axis=0) < 0).sum())
    pulse_drops = int((np.diff(prediction_table.values, axis=1) < 0).sum())
    consistency = pd.DataFrame(
        [
            {
                "direction": "increasing voltage",
                "comparisons": voltage_comparisons,
                "local_drops": voltage_drops,
            },
            {
                "direction": "increasing pulses",
                "comparisons": pulse_comparisons,
                "local_drops": pulse_drops,
            },
        ]
    )
    consistency["non_decreasing_fraction"] = 1 - (
        consistency["local_drops"] / consistency["comparisons"]
    )
    return consistency, grid


def plot_prediction_surface(
    final_model: RandomForestRegressor,
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    voltage_grid, pulse_grid = np.meshgrid(
        np.linspace(data["voltage_V"].min(), data["voltage_V"].max(), 141),
        np.linspace(data["pulses"].min(), data["pulses"].max(), 141),
    )
    points = pd.DataFrame(
        {"voltage_V": voltage_grid.ravel(), "pulses": pulse_grid.ravel()}
    )
    depth = final_model.predict(points).reshape(voltage_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    contour = ax.contourf(voltage_grid, pulse_grid, depth, levels=20, cmap="viridis")
    ax.scatter(data["voltage_V"], data["pulses"], s=8, c="white", alpha=0.5)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Number of pulses")
    ax.set_title("Random forest prediction surface")
    fig.colorbar(contour, ax=ax, label="Predicted depth (μm)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def portable_path(path: Path) -> str:
    """Use repository-relative paths in tracked summaries when possible."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the complete workflow and return its concise run summary."""
    configure_plots()
    result_dir, figure_dir = prepare_output_directories(args.output_root)
    data, data_quality = load_and_validate_data(args.data)
    data_quality.to_csv(result_dir / "data_quality.csv", index=False)

    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.random_state
    )
    LOGGER.info("Loaded %d observations from %s", len(data), args.data.resolve())
    LOGGER.info("Split data into %d training and %d test observations", len(x_train), len(x_test))

    models = make_models(args.random_state, args.n_jobs)
    cv = make_cv(args.random_state)
    cv_results = compare_models(models, x_train, y_train, cv)
    cv_results.to_csv(result_dir / "cv_model_comparison.csv", index=False)
    plot_cv_comparison(cv_results, figure_dir / "cv_model_comparison.png")

    LOGGER.info("Searching 48 random-forest hyperparameter combinations")
    final_model, tuning_results, best_cv_rmse = tune_random_forest(
        x_train, y_train, cv, args.random_state, args.n_jobs
    )
    tuning_results.to_csv(result_dir / "tuning_results.csv", index=False)
    LOGGER.info("Best parameters: %s", final_model.get_params())
    LOGGER.info("Best training CV RMSE: %.4f μm", best_cv_rmse)

    model_metrics = evaluate_models(
        models, final_model, x_train, x_test, y_train, y_test
    )
    model_metrics.to_csv(result_dir / "model_metrics.csv", index=False)

    predictions, test_results = make_test_predictions(final_model, x_test, y_test)
    test_results.to_csv(result_dir / "test_predictions.csv", index=False)
    plot_test_diagnostics(y_test, predictions, figure_dir)

    importance = calculate_feature_importance(
        final_model, x_test, y_test, args.random_state
    )
    importance.to_csv(result_dir / "feature_importance.csv", index=False)
    plot_feature_importance(importance, figure_dir / "feature_importance.png")

    group_results = grouped_validation(final_model, x, y, data)
    group_results.to_csv(result_dir / "grouped_validation.csv", index=False)

    consistency, prediction_grid = physical_consistency(final_model, data)
    consistency.to_csv(result_dir / "physical_consistency.csv", index=False)
    prediction_grid.to_csv(result_dir / "prediction_grid.csv", index=False)
    plot_prediction_surface(final_model, data, figure_dir / "prediction_surface.png")

    tuned_test = model_metrics.loc[
        (model_metrics["model"] == "Tuned random forest")
        & (model_metrics["split"] == "test")
    ].iloc[0]
    summary: dict[str, Any] = {
        "data_path": portable_path(args.data),
        "observations": len(data),
        "training_observations": len(x_train),
        "test_observations": len(x_test),
        "random_state": args.random_state,
        "best_parameters": {
            "n_estimators": final_model.n_estimators,
            "max_depth": final_model.max_depth,
            "min_samples_leaf": final_model.min_samples_leaf,
            "max_features": final_model.max_features,
        },
        "best_training_cv_RMSE_um": best_cv_rmse,
        "test_R2": tuned_test["R2"],
        "test_MAE_um": tuned_test["MAE_um"],
        "test_RMSE_um": tuned_test["RMSE_um"],
        "results_directory": portable_path(result_dir),
        "figures_directory": portable_path(figure_dir),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (result_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=json_safe)

    LOGGER.info(
        "Held-out test | R² %.4f | MAE %.4f μm | RMSE %.4f μm",
        tuned_test["R2"],
        tuned_test["MAE_um"],
        tuned_test["RMSE_um"],
    )
    LOGGER.info("Saved results to %s and figures to %s", result_dir, figure_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        run_workflow(args)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.error("Workflow stopped: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
