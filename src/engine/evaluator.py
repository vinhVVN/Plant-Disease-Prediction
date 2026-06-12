import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_loader, class_names, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    
    # Save CSV
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(results_dir, "classification_report.csv")
    report_df.to_csv(metrics_csv_path)

    print(report_df)

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    save_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nĐã lưu confusion matrix tại: {save_path}")
