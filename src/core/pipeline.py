from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer


def build_pipeline(config):

    # 🔥 AUTO detect will be passed from train
    num_cols = config.get("num_cols", [])
    cat_cols = config.get("cat_cols", [])

    # -------------------------
    # NUMERIC PIPELINE
    # -------------------------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # -------------------------
    # CATEGORICAL PIPELINE
    # -------------------------
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # -------------------------
    # COMBINE
    # -------------------------
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # -------------------------
    # FINAL PIPELINE
    # -------------------------
    pipeline = Pipeline([
        ("preprocessing", preprocessor),

        ("smote", SMOTE(random_state=42)),

        ("pca", PCA(
            n_components=config.get("pca_components", 10)
        )),

        ("svm", SVC(
            kernel=config.get("kernel", "rbf"),
            C=config.get("C", 1.0),
            gamma=config.get("gamma", "scale"),
            probability=True
        ))
    ])

    return pipeline