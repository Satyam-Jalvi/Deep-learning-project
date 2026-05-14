from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def build_pipeline():

    # numerical columns (all are numeric here)
    num_cols = [
        'age','sex','cp','trestbps','chol',
        'fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal'
    ]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('pca', PCA(n_components=8)),
        ('svm', SVC(kernel='rbf', probability=True))
    ])

    return pipeline