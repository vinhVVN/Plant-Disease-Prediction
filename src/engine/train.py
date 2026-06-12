import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.metrics import MetricTracker

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.epochs = config['training']['epochs']
        self.patience = config['training']['early_stopping_patience']
        self.save_dir = config['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        
        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']
        betas = tuple(config['training'].get('betas', (0.9, 0.999)))
        
        opt_name = config['training'].get('optimizer', 'Adam')
        if opt_name == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif opt_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=betas)
            
        self.tracker = MetricTracker()

    def train(self):
        best_val_f1 = 0.0
        epochs_no_improve = 0
        history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            self.tracker.reset()
            train_pbar = tqdm(self.train_loader, desc="Training")
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                preds = torch.argmax(outputs, dim=1)
                self.tracker.update(labels, preds, loss.item())
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            train_metrics = self.tracker.compute()
            
            # Validation phase
            self.model.eval()
            self.tracker.reset()
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc="Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    self.tracker.update(labels, preds, loss.item())
                    
            val_metrics = self.tracker.compute()
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['macro_f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['macro_f1'])

            # Checkpointing & Early Stopping
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                epochs_no_improve = 0
                best_model_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"*** Saved best model with Macro F1: {best_val_f1:.4f} ***")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        return history
