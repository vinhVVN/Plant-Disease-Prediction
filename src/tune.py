import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import optuna

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_hpo_transforms, get_baseline_transforms
from src.models.mobilenet_v3 import MobileNetV3Small
from src.utils.weight_loader import migrate_imagenet_weights
from src.utils.metrics import MetricTracker

def objective(trial):
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_dir = config['data']['dataset_dir']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    
    # Isolation Strategy: HPO Transform for training, Baseline for validation
    train_transform = get_hpo_transforms(image_size)
    val_transform = get_baseline_transforms(image_size)
    
    # Safe subset splitting to prevent transform leakage
    train_dataset_full = PlantVillageDataset(dataset_dir, transform=train_transform)
    val_dataset_full = PlantVillageDataset(dataset_dir, transform=val_transform)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset_full), generator=generator).tolist()
    train_size = int(0.8 * len(train_dataset_full))
    
    train_dataset = Subset(train_dataset_full, indices[:train_size])
    val_dataset = Subset(val_dataset_full, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['data']['num_workers'])

    model = MobileNetV3Small(num_classes=config['model']['num_classes'])
    if config['model']['pretrained']:
        model = migrate_imagenet_weights(model, num_classes=config['model']['num_classes'])
    model = model.to(device)

    # Search Space Configuration
    lr = trial.suggest_float("lr", float(config['hpo']['lr_min']), float(config['hpo']['lr_max']), log=True)
    weight_decay = trial.suggest_float("weight_decay", float(config['hpo']['weight_decay_min']), float(config['hpo']['weight_decay_max']), log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    tracker = MetricTracker()
    
    # Shortened training loop for rapid trials
    epochs = 5 
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        tracker.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                tracker.update(labels, preds)
                
        val_metrics = tracker.compute()
        val_f1 = val_metrics['macro_f1']
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_f1

def tune_hyperparameters():
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config['hpo']['n_trials'])
    
    print("\n Optuna Tuning Completed")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (Macro F1): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best params back to config
    config['training']['learning_rate'] = trial.params['lr']
    config['training']['weight_decay'] = trial.params['weight_decay']
    config['training']['optimizer'] = trial.params['optimizer']
    
    with open("configs/default_config.yaml", 'w') as f:
        yaml.safe_dump(config, f)
        
    print("Successfully updated configs/default_config.yaml with best hyperparameters.")

if __name__ == '__main__':
    tune_hyperparameters()
