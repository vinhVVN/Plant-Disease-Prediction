import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_baseline_transforms
from src.models.mobilenet_v3 import MobileNetV3Small
from src.utils.weight_loader import migrate_imagenet_weights
from src.utils.metrics import MetricTracker

def train_model(config_path="configs/default_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_dir = config['data']['dataset_dir']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    
    # Use Baseline Transforms for Academic Control
    transform = get_baseline_transforms(image_size)
    full_dataset = PlantVillageDataset(dataset_dir, transform=transform)
    
    # Reproducible split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_size = int(0.8 * len(full_dataset))

    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['data']['num_workers'])

    # Model Initialization & Strict Surgery
    model = MobileNetV3Small(num_classes=config['model']['num_classes'])
    if config['model']['pretrained']:
        model = migrate_imagenet_weights(model, num_classes=config['model']['num_classes'])
    model = model.to(device)

    # Training Engine Essentials
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'],
                           betas=tuple(config['training']['betas']))

    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    tracker = MetricTracker()

    # Epoch Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        tracker.reset()
        
        train_pbar = tqdm(train_loader, desc="Training")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            tracker.update(labels, preds, loss.item())
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_metrics = tracker.compute()
        
        # Validation Loop
        model.eval()
        tracker.reset()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                tracker.update(labels, preds, loss.item())
                
        val_metrics = tracker.compute()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")

        # Checkpointing & Early Stopping (Optimizing for Macro F1)
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"*** Saved best model with Macro F1: {best_val_f1:.4f} ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == '__main__':
    train_model()
