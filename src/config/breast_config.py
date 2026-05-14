CONFIG = {
    "name": "breast",

    "data_path": "data/breast_cancer/data.csv",
    "target": "diagnosis",

    # optional (pipeline defaults will work even if removed)
    "pca_components": 10,
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",

    "test_size": 0.2,
    "stratify": True
}