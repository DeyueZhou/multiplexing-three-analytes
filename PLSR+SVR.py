from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Parameter 1: spectral data source. Options: "raw", "als", "both".
DATA_MODE = "raw"
RAW_DATA_PATH = Path("dataset-raw.csv")
ALS_DATA_PATH = Path("dataset-ALS-processed.csv")

# Parameter 2: repeated spectra. Options: "individual", "average".
SPECTRUM_MODE = "average"
INDIVIDUAL_SPECTRAL_NUMBERS = (1, 2, 3)
AVERAGE_SPECTRAL_GROUPS = (tuple(range(1, 16)),)

# Parameter 3: substrate selection. Use None for all substrates.
SELECTED_SUBSTRATES = (4, 5, 6, 7)

INCLUDE_SUBSTRATE_NUMBER = True
TEST_SIZE = 0.2
CV_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_PREFIX = "PLSR_SVR"
OUTPUT_DIR = Path(".")
N_JOBS = -1

TARGET_COLUMNS = [
    "UA concentration",
    "Adenine concentration",
    "Cocaine concentration",
]

META_COLUMNS = [
    "mixture",
    "UA concentration",
    "Adenine concentration",
    "Cocaine concentration",
    "substrate number",
    "spectral number",
]

PARAM_GRID = {
    "plsr__n_components": [5, 10, 15, 20, 25, 30, 40, 50],
    "svr__regressor__estimator__C": [0.1, 1, 10, 100, 1000],
    "svr__regressor__estimator__gamma": ["scale", 1e-4, 1e-3, 1e-2, 1e-1, 1],
    "svr__regressor__estimator__epsilon": [0.001, 0.01, 0.1, 1],
}


class PLSRFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20, raman_columns=None, include_substrate=True):
        self.n_components = n_components
        self.raman_columns = raman_columns
        self.include_substrate = include_substrate

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.raman_columns_ = list(self.raman_columns)
        self.raman_scaler_ = StandardScaler()
        X_raman = self.raman_scaler_.fit_transform(X[self.raman_columns_])

        self.n_components_ = min(self.n_components, X_raman.shape[0] - 1, X_raman.shape[1])
        self.pls_ = PLSRegression(n_components=self.n_components_, scale=False)
        self.pls_.fit(X_raman, y)

        if self.include_substrate:
            self.substrate_scaler_ = StandardScaler().fit(X[["substrate number"]])
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_raman = self.raman_scaler_.transform(X[self.raman_columns_])
        X_pls = self.pls_.transform(X_raman)

        if not self.include_substrate:
            return X_pls

        X_substrate = self.substrate_scaler_.transform(X[["substrate number"]])
        return np.hstack([X_pls, X_substrate])


def read_spectral_table(path):
    df = pd.read_csv(path)
    raman_columns = [col for col in df.columns if col not in META_COLUMNS]
    return df, raman_columns


def load_input_data():
    if DATA_MODE == "raw":
        return read_spectral_table(RAW_DATA_PATH)
    if DATA_MODE == "als":
        return read_spectral_table(ALS_DATA_PATH)
    if DATA_MODE != "both":
        raise ValueError("DATA_MODE must be 'raw', 'als', or 'both'.")

    raw_df, raw_cols = read_spectral_table(RAW_DATA_PATH)
    als_df, als_cols = read_spectral_table(ALS_DATA_PATH)
    if not raw_df[META_COLUMNS].reset_index(drop=True).equals(als_df[META_COLUMNS].reset_index(drop=True)):
        raise ValueError("Raw and ALS metadata must match row-by-row.")

    df = raw_df[META_COLUMNS].copy()
    raw_features = raw_df[raw_cols].add_prefix("raw__")
    als_features = als_df[als_cols].add_prefix("als__")
    return pd.concat([df, raw_features, als_features], axis=1), list(raw_features.columns) + list(als_features.columns)


def prepare_dataset(df, raman_columns):
    df = df.copy()
    df["substrate number"] = pd.to_numeric(df["substrate number"], errors="coerce")
    df["spectral number"] = pd.to_numeric(df["spectral number"], errors="coerce")
    df = df.dropna(subset=["substrate number", "spectral number"])

    if SELECTED_SUBSTRATES is not None:
        df = df[df["substrate number"].isin(SELECTED_SUBSTRATES)].copy()

    df["condition_group"] = (
        df["mixture"].astype(str)
        + "_substrate_"
        + df["substrate number"].astype(int).astype(str)
    )

    if SPECTRUM_MODE == "individual":
        selected = sorted(INDIVIDUAL_SPECTRAL_NUMBERS)
        df = df[df["spectral number"].isin(selected)].copy()
        df["spectrum_type"] = label_numbers(selected, "individual")
        df["n_spectra_averaged"] = 1
        df["used_spectral_numbers"] = df["spectral number"].astype(int).astype(str)
        return df.reset_index(drop=True)

    if SPECTRUM_MODE != "average":
        raise ValueError("SPECTRUM_MODE must be 'individual' or 'average'.")

    rows = []
    group_columns = META_COLUMNS[:-1] + ["condition_group"]
    for _, group in df.groupby(group_columns, sort=False):
        for i, spectral_group in enumerate(AVERAGE_SPECTRAL_GROUPS, start=1):
            selected = sorted(spectral_group)
            block = group[group["spectral number"].isin(selected)]
            if block.empty:
                continue

            row = group.iloc[0][META_COLUMNS[:-1] + ["condition_group"]].to_dict()
            found = sorted(block["spectral number"].astype(int).unique())
            row.update(
                {
                    "spectral number": f"avg_{i}",
                    "spectrum_type": label_numbers(selected, "average"),
                    "n_spectra_averaged": len(block),
                    "used_spectral_numbers": ",".join(map(str, found)),
                }
            )
            row.update(block[raman_columns].mean().to_dict())
            rows.append(row)

    if not rows:
        raise ValueError("No averaged spectra were generated.")
    return pd.DataFrame(rows)


def split_dataset(df, raman_columns):
    feature_columns = list(raman_columns)
    if INCLUDE_SUBSTRATE_NUMBER:
        feature_columns.append("substrate number")

    X = df[feature_columns]
    y = df[TARGET_COLUMNS]
    groups = df["condition_group"]
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        ).split(X, y, groups=groups)
    )

    meta_columns = [
        "mixture",
        "substrate number",
        "spectral number",
        "condition_group",
        "spectrum_type",
        "n_spectra_averaged",
        "used_spectral_numbers",
    ]
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        groups.iloc[train_idx],
        df.iloc[test_idx][meta_columns].copy(),
    )


def build_model(raman_columns):
    svr = MultiOutputRegressor(SVR(kernel="rbf"))
    return Pipeline(
        [
            ("plsr", PLSRFeatureExtractor(raman_columns=raman_columns, include_substrate=INCLUDE_SUBSTRATE_NUMBER)),
            ("svr", TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())),
        ]
    )


def evaluate(y_true, y_pred):
    rows = [metric_row("Overall", y_true, y_pred)]
    rows += [metric_row(target, y_true[:, i], y_pred[:, i]) for i, target in enumerate(TARGET_COLUMNS)]
    return pd.DataFrame(rows)


def metric_row(target, y_true, y_pred):
    return {
        "target": target,
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run_analysis():
    df, raman_columns = load_input_data()
    df = prepare_dataset(df, raman_columns)
    X_train, X_test, y_train, y_test, groups_train, meta_test = split_dataset(df, raman_columns)

    cv_splits = min(CV_SPLITS, groups_train.nunique())
    grid = GridSearchCV(
        build_model(raman_columns),
        param_grid=PARAM_GRID,
        scoring="r2",
        cv=GroupKFold(n_splits=cv_splits),
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid.fit(X_train, y_train, groups=groups_train)

    y_pred = grid.best_estimator_.predict(X_test)
    metrics = evaluate(y_test.to_numpy(), y_pred)
    save_results(meta_test, y_test.to_numpy(), y_pred, metrics, grid)

    print("Best parameters:", grid.best_params_)
    print(f"Best CV R2: {grid.best_score_:.4f}")
    print(metrics.to_string(index=False))


def save_results(meta_test, y_true, y_pred, metrics, grid):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    prefix = output_prefix()

    predictions = meta_test.reset_index(drop=True)
    for i, target in enumerate(TARGET_COLUMNS):
        predictions[f"{target}_true"] = y_true[:, i]
        predictions[f"{target}_pred"] = y_pred[:, i]
        predictions[f"{target}_error"] = y_pred[:, i] - y_true[:, i]

    metrics = metrics.copy()
    metrics["best_CV_R2"] = grid.best_score_
    metrics["best_params"] = str(grid.best_params_)
    predictions.to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / f"{prefix}_metrics.csv", index=False)


def output_prefix():
    substrates = "all" if SELECTED_SUBSTRATES is None else label_numbers(SELECTED_SUBSTRATES)
    spectra = (
        label_numbers(INDIVIDUAL_SPECTRAL_NUMBERS, "individual")
        if SPECTRUM_MODE == "individual"
        else "__".join(label_numbers(group, "avg") for group in AVERAGE_SPECTRAL_GROUPS)
    )
    return f"{OUTPUT_PREFIX}_{DATA_MODE}_substrates_{substrates}_{spectra}"


def label_numbers(numbers, prefix=None):
    numbers = sorted(map(int, numbers))
    label = (
        f"{numbers[0]}_to_{numbers[-1]}"
        if numbers == list(range(numbers[0], numbers[-1] + 1))
        else "_".join(map(str, numbers))
    )
    return f"{prefix}_{label}" if prefix else label


if __name__ == "__main__":
    run_analysis()
