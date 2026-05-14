import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_diabetes_data(df, config):

    save_dir = f"reports/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)

    sns.countplot(x=df[config["target"]])
    plt.title("Diabetes Distribution")
    plt.savefig(f"{save_dir}/target_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{save_dir}/correlation.png")
    plt.close()

    print("✅ Diabetes visualization saved")