import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dermatology_data(df, config):

    # clean columns
    df.columns = df.columns.str.strip()

    # save directory
    save_dir = f"reports/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------
    # TARGET DISTRIBUTION
    # ---------------------------------
    plt.figure(figsize=(8, 5))

    sns.countplot(x=df[config["target"]])

    plt.title("Dermatology Class Distribution")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.savefig(f"{save_dir}/target_distribution.png")

    plt.close()

    # ---------------------------------
    # CORRELATION HEATMAP
    # ---------------------------------
    plt.figure(figsize=(16, 14))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm"
    )

    plt.title("Dermatology Correlation Heatmap")

    plt.savefig(f"{save_dir}/correlation_heatmap.png")

    plt.close()

    # ---------------------------------
    # AGE DISTRIBUTION
    # ---------------------------------
    plt.figure(figsize=(8, 5))

    sns.histplot(df["age"], bins=20, kde=True)

    plt.title("Age Distribution")

    plt.xlabel("Age")

    plt.ylabel("Frequency")

    plt.savefig(f"{save_dir}/age_distribution.png")

    plt.close()

    # ---------------------------------
    # FEATURE DISTRIBUTION
    # ---------------------------------
    top_features = [

        "erythema",

        "scaling",

        "itching",

        "acanthosis",

        "hyperkeratosis"
    ]

    for feature in top_features:

        plt.figure(figsize=(7, 5))

        sns.countplot(x=df[feature])

        plt.title(f"{feature} Distribution")

        plt.savefig(f"{save_dir}/{feature}_distribution.png")

        plt.close()

    print(f"✅ Dermatology plots saved in: {save_dir}")