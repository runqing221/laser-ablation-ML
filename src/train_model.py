#!/usr/bin/env python3
"""Reproduce the laser-ablation depth prediction results.

This script is a compact, non-interactive version of the project notebook. It
compares baseline models, tunes a random forest, evaluates a held-out test set,
and saves the main tables and figures used in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ablation_data.csv"
FEATURES = ["voltage_V", "pulses"]
TARGET = "depth_um"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT,
        help="Save results/ and figures/ under this directory.",
    )
    return parser.parse_args()


def regression_metrics(y_true, y_pred):
    """Return the three regression metrics reported in the project."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE_um": mean_absolute_error(y_true, y_pred),
        "RMSE_um": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    figure_dir = args.output_root / "figures"
    result_dir = args.output_root / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and check the tidy experimental data.
    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + [TARGET]
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"Required columns: {required_columns}")
    if data[required_columns].isna().any().any():
        raise ValueError("The modelling data contain missing values.")
    if data.duplicated(FEATURES).any():
        raise ValueError("Duplicate voltage-pulse settings were found.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Data: {len(data)} rows | train: {len(X_train)} | test: {len(X_test)}")

    # 2. Compare transparent baselines before tuning the final model.
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Quadratic regression": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        ),
        "Reference random forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_rows.append(
            {
                "model": name,
                "R2_mean": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "MAE_mean_um": -scores["test_mae"].mean(),
                "RMSE_mean_um": -scores["test_rmse"].mean(),
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values("RMSE_mean_um")
    cv_results.to_csv(result_dir / "cv_model_comparison.csv", index=False)

    ordered = cv_results.sort_values("RMSE_mean_um", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(ordered["model"], ordered["RMSE_mean_um"], color="#3b82b8")
    ax.set_xlabel("Mean cross-validation RMSE (μm)")
    ax.set_title("Model comparison on the training set")
    save_figure(fig, figure_dir / "cv_model_comparison.png")

    # 3. Tune the forest using the training set only (48 combinations).
    parameter_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt"],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    print("Best parameters:", search.best_params_)
    print(f"Best CV RMSE: {-search.best_score_:.4f} μm")

    # 4. Fit each model and evaluate the untouched test set once.
    metric_rows = []
    for name, model in {**models, "Tuned random forest": final_model}.items():
        model.fit(X_train, y_train)
        for split, split_X, split_y in [
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ]:
            metric_rows.append(
                {
                    "model": name,
                    "split": split,
                    **regression_metrics(split_y, model.predict(split_X)),
                }
            )

    model_metrics = pd.DataFrame(metric_rows)
    model_metrics.to_csv(result_dir / "model_metrics.csv", index=False)

    test_prediction = final_model.predict(X_test)
    test_results = X_test.copy()
    test_results["measured_depth_um"] = y_test
    test_results["predicted_depth_um"] = test_prediction
    test_results["residual_um"] = y_test - test_prediction
    test_results["absolute_error_um"] = test_results["residual_um"].abs()
    test_results.sort_values("absolute_error_um", ascending=False).to_csv(
        result_dir / "test_predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(y_test, test_prediction, s=45, alpha=0.8)
    limits = [
        min(y_test.min(), test_prediction.min()) - 0.5,
        max(y_test.max(), test_prediction.max()) + 0.5,
    ]
    ax.plot(limits, limits, "--", color="black", lw=1)
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Measured depth (μm)",
        ylabel="Predicted depth (μm)",
        title="Measured and predicted depth",
    )
    save_figure(fig, figure_dir / "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(test_prediction, y_test - test_prediction, s=45, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set(
        xlabel="Predicted depth (μm)",
        ylabel="Residual: measured - predicted (μm)",
        title="Test-set residuals",
    )
    save_figure(fig, figure_dir / "residuals.png")

    # 5. Explain which input matters most.
    permutation = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="r2",
    )
    positive = np.maximum(permutation.importances_mean, 0)
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "impurity_fraction": final_model.feature_importances_,
            "permutation_R2_drop": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
            "permutation_fraction": positive / positive.sum(),
        }
    )
    importance.to_csv(result_dir / "feature_importance.csv", index=False)

    plot_data = importance.set_index("feature")[
        ["impurity_fraction", "permutation_fraction"]
    ]
    ax = plot_data.plot(
        kind="bar", figsize=(6.5, 4.5), rot=0, color=["#3b82b8", "#e58b3a"]
    )
    ax.set_ylabel("Relative importance")
    ax.set_title("Feature importance")
    ax.legend(["Tree impurity", "Test permutation"])
    save_figure(ax.get_figure(), figure_dir / "feature_importance.png")

    # 6. Test harder cases where complete voltage or pulse levels are unseen.
    group_rows = []
    for label, groups, folds in [
        ("Unseen voltage levels", data["voltage_V"], 4),
        ("Unseen pulse levels", data["pulses"], 5),
    ]:
        scores = cross_validate(
            final_model,
            X,
            y,
            groups=groups,
            cv=GroupKFold(n_splits=folds),
            scoring=scoring,
        )
        group_rows.append(
            {
                "check": label,
                "R2_mean": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "MAE_mean_um": -scores["test_mae"].mean(),
                "RMSE_mean_um": -scores["test_rmse"].mean(),
            }
        )
    pd.DataFrame(group_rows).to_csv(
        result_dir / "grouped_validation.csv", index=False
    )

    # 7. Check physical trend consistency on the measured parameter grid.
    voltage_values = np.sort(data["voltage_V"].unique())
    pulse_values = np.sort(data["pulses"].unique())
    experimental_grid = pd.MultiIndex.from_product(
        [voltage_values, pulse_values], names=FEATURES
    ).to_frame(index=False)
    experimental_grid["predicted_depth_um"] = final_model.predict(
        experimental_grid[FEATURES]
    )
    prediction_table = experimental_grid.pivot(
        index="voltage_V", columns="pulses", values="predicted_depth_um"
    )
    consistency = pd.DataFrame(
        [
            {
                "direction": "increasing voltage",
                "comparisons": 105,
                "local_drops": int(
                    (np.diff(prediction_table.values, axis=0) < 0).sum()
                ),
            },
            {
                "direction": "increasing pulses",
                "comparisons": 112,
                "local_drops": int(
                    (np.diff(prediction_table.values, axis=1) < 0).sum()
                ),
            },
        ]
    )
    consistency["non_decreasing_fraction"] = 1 - (
        consistency["local_drops"] / consistency["comparisons"]
    )
    consistency.to_csv(result_dir / "physical_consistency.csv", index=False)

    voltage_grid, pulse_grid = np.meshgrid(
        np.linspace(460, 600, 141), np.linspace(10, 150, 141)
    )
    surface_points = pd.DataFrame(
        {"voltage_V": voltage_grid.ravel(), "pulses": pulse_grid.ravel()}
    )
    surface_depth = final_model.predict(surface_points).reshape(voltage_grid.shape)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    contour = ax.contourf(
        voltage_grid, pulse_grid, surface_depth, levels=20, cmap="viridis"
    )
    ax.scatter(data["voltage_V"], data["pulses"], s=8, c="white", alpha=0.5)
    ax.set(
        xlabel="Voltage (V)",
        ylabel="Number of pulses",
        title="Random forest prediction surface",
    )
    fig.colorbar(contour, ax=ax, label="Predicted depth (μm)")
    save_figure(fig, figure_dir / "prediction_surface.png")

    final_test = model_metrics.query(
        "model == 'Tuned random forest' and split == 'test'"
    ).iloc[0]
    print(
        "Test result: "
        f"R²={final_test['R2']:.4f}, "
        f"MAE={final_test['MAE_um']:.4f} μm, "
        f"RMSE={final_test['RMSE_um']:.4f} μm"
    )
    print(f"Saved results to {result_dir} and figures to {figure_dir}")


if __name__ == "__main__":
    main()
