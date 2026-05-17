import pandas as pd

from src.config.heart_config import CONFIG as HEART_CONFIG
from src.config.breast_config import CONFIG as BREAST_CONFIG
from src.config.diabetes_config import CONFIG as DIABETES_CONFIG
from src.config.dermatology_config import CONFIG as DERMATOLOGY_CONFIG

from src.core.train import train
from src.core.evaluate import evaluate

from src.visualization.heart_plots import plot_heart_data
from src.visualization.breast_plots import plot_breast_data
from src.visualization.diabetes_plots import plot_diabetes_data
from src.visualization.dermatology_plots import plot_dermatology_data


def run_pipeline(config, plot_func):

    print(f"\n🔹 Processing: {config['name']}")

    # load data
    # special handling for dermatology dataset
    if config["name"] == "dermatology":

        df = pd.read_csv(
            config["data_path"],
            na_values=["?"]
        )

    else:

        df = pd.read_csv(config["data_path"])

    # visualization
    plot_func(df, config)

    # train
    model, X_test, y_test = train(config)

    # evaluate
    evaluate(model, X_test, y_test, config)


def main():

    print("\n====== Training All Models ======")

    pipelines = [
        (HEART_CONFIG, plot_heart_data),
        (BREAST_CONFIG, plot_breast_data),
        (DIABETES_CONFIG, plot_diabetes_data),
        (DERMATOLOGY_CONFIG, plot_dermatology_data) 
    ]

    for config, plot_func in pipelines:
        run_pipeline(config, plot_func)

    print("\n✅ All models trained and saved successfully!")


if __name__ == "__main__":
    main()