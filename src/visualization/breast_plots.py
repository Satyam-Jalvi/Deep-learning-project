import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_breast_data(df, config):

    df.columns = df.columns.str.strip()

    save_dir = f"reports/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)

    # target distribution
    sns.countplot(x=df[config["target"]])
    plt.title("Diagnosis Distribution (M vs B)")
    plt.savefig(f"{save_dir}/target_distribution.png")
    plt.close()

    # correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{save_dir}/correlation.png")
    plt.close()

    print("✅ Breast visualization saved")