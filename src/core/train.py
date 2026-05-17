import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from src.core.pipeline import build_pipeline


def train(config):

    print(f"\n🔹 Training: {config['name']}")

    df = pd.read_csv(config["data_path"])

    # -------------------------
    # CLEAN COLUMNS
    # -------------------------
    df.columns = df.columns.str.strip()

    # remove useless columns
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    # replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # try converting object columns to numeric safely
    for col in df.columns:

        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # -------------------------
    # TARGET
    # -------------------------
    target = config["target"]

    # encode target if needed
    if df[target].dtype == "object":

        unique_vals = set(df[target].dropna().unique())

        if unique_vals == {"M", "B"}:
            df[target] = df[target].map({"B": 0, "M": 1})

        elif unique_vals == {"Yes", "No"}:
            df[target] = df[target].map({"No": 0, "Yes": 1})

    X = df.drop(target, axis=1)

    y = df[target]

    # -------------------------
    # AUTO DETECT FEATURES
    # -------------------------
    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # override if config provides
    num_cols = config.get("num_cols", num_cols)

    cat_cols = config.get("cat_cols", cat_cols)

    config["num_cols"] = num_cols

    config["cat_cols"] = cat_cols

    # -------------------------
    # TRAIN TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=config.get("test_size", 0.2),

        random_state=42,

        stratify=y if config.get("stratify", True) else None
    )

    # -------------------------
    # BUILD PIPELINE
    # -------------------------
    model = build_pipeline(config)

    # -------------------------
    # GRID SEARCH
    # -------------------------
    param_grid = {

        "svm__C": [0.1, 1, 10],

        "svm__kernel": ["linear", "rbf"],

        "svm__gamma": ["scale", "auto"],

        "pca__n_components": [5, 10]
    }

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=5,

        scoring="accuracy",

        n_jobs=-1,

        verbose=2
    )

    # -------------------------
    # TRAIN
    # -------------------------
    grid_search.fit(X_train, y_train)

    # best model
    model = grid_search.best_estimator_

    print("\n✅ Best Parameters:")

    print(grid_search.best_params_)

    print("\n✅ Best CV Score:")

    print(grid_search.best_score_)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    os.makedirs("models", exist_ok=True)

    path = f"models/{config['name']}_model.pkl"

    joblib.dump(model, path)

    print(f"✅ Model saved: {path}")

    return model, X_test, y_test