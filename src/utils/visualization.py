import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def plot_learning_curves(history_dict, save_path):
    """
    Plots Training vs Validation curves for Loss, Accuracy, and Macro-F1.
    Replicates Figures 5 & 6 from the Khan 2023 paper.
    """
    epochs = range(1, len(history_dict['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss Plot
    axes[0].plot(epochs, history_dict['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs, history_dict['val_loss'], label='Val Loss', color='orange')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy Plot
    axes[1].plot(epochs, history_dict['train_acc'], label='Train Accuracy', color='green')
    axes[1].plot(epochs, history_dict['val_acc'], label='Val Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Macro F1 Plot
    axes[2].plot(epochs, history_dict['train_f1'], label='Train Macro-F1', color='purple')
    axes[2].plot(epochs, history_dict['val_f1'], label='Val Macro-F1', color='brown')
    axes[2].set_title('Training and Validation Macro-F1')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Macro-F1')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Renders a high-quality heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def extract_features_and_tsne(model, dataloader, device, save_path):
    """
    Extracts 1024-dimensional latent features from the penultimate layer
    and plots a 2D t-SNE scatter plot color-coded by class (Replicating Figure 7).
    """
    features = []
    labels = []
    
    # Define a hook to intercept the output of the 1024-D layer 
    # (using Hswish activation, model.classifier[1], to get the post-activation 1024-D features)
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())
        
    # Register the forward hook
    handle = model.classifier[1].register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)
            labels.extend(lbls.numpy())
            
    # Remove the hook after extraction to maintain architectural integrity
    handle.remove()
    
    # Concatenate all extracted features from batches
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    print("Running t-SNE dimensionality reduction... This may take a while.")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # Render clustered 2D scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab20', alpha=0.7, s=15)
    plt.title('t-SNE Visualization of 1024-D Latent Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Class Index')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE plot successfully saved to {save_path}")
