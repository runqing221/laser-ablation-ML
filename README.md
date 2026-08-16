# Random Forest Prediction of Laser Ablation Depth

This project rebuilds the random forest regression part of my undergraduate laser ablation study. The dataset contains 120 measurements from nanosecond laser single-point ablation experiments on 6061 aluminum alloy.

The model uses two experimental settings:

- input voltage: 460–600 V
- number of pulses: 10–150

The target is the measured maximum ablation depth in micrometres. Laser power is not added as a separate feature because it is calculated directly from voltage.

## Method

The data is divided into an 80% training set and a 20% test set with random seed 42. Mean, linear and quadratic regression models are used as references. Random forest parameters are selected with five-fold cross-validation on the training set, followed by one evaluation on the held-out test set.

Additional checks include grouped validation with complete voltage or pulse levels withheld, permutation feature importance, residual analysis and monotonicity checks over the experimental grid.

Random forest inputs are not standardised because tree splits are not affected by feature scale.

## Main results

The tuned random forest obtained:

- test R²: 0.9814
- test MAE: 0.6373 μm
- test RMSE: 0.7864 μm

The reproduced result is close to the thesis result but not exactly identical. The thesis records the train-test split seed but does not state the random forest seed or software version; this version fixes both for reproducibility.

Pulse count is the main predictor. The stricter grouped validation scores are lower than the random holdout result, showing that prediction at completely unseen voltage or pulse levels is more difficult.

The model is only intended for interpolation within 460–600 V and 10–150 pulses.

## Files

- `notebooks/random_forest_analysis.ipynb`: complete analysis
- `data/ablation_data.xlsx`: original measurement table
- `data/ablation_data.csv`: tidy data used by the notebook
- `figures/`: model comparison and result plots
- `results/`: evaluation tables and test predictions

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Start Jupyter from the repository root and open `notebooks/random_forest_analysis.ipynb`.
