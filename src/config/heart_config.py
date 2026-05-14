CONFIG = {
    "name": "heart",

    "data_path": "data/heart_disease/data.csv",
    "target": "AHD",

    # numeric columns
    "num_cols": [
        "Age", "RestBP", "Chol", "MaxHR",
        "Oldpeak", "Ca"
    ],

    # categorical columns
    "cat_cols": [
        "Sex", "ChestPain", "Fbs", "RestECG",
        "ExAng", "Slope", "Thal"
    ],

    "pca_components": 8,
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",

    "test_size": 0.2,
    "stratify": True
}