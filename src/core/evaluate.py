import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

def evaluate(model, X_test, y_test, config):

    print(f"\n🔹 Evaluating: {config['name']}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print(report)

    save_dir = f"reports/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

    # ROC curve
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()

        plt.savefig(f"{save_dir}/roc_curve.png")
        plt.close()
    except:
        print("ROC not available")

    print("✅ Evaluation saved")