import joblib
import pandas as pd

def predict(config, input_json):

    model = joblib.load(f"models/{config['name']}_model.pkl")

    df = pd.DataFrame([input_json])

    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df)[0].max()
        return pred, prob

    return pred, None