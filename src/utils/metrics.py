import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class MetricTracker:
    """
    Utility class to track losses and compute evaluation metrics.
    Crucial for handling the imbalanced nature of PlantVillage using Macro F1-Score.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.losses = []

    def update(self, y_true, y_pred, loss=None):
        self.y_true.extend(y_true.cpu().numpy())
        self.y_pred.extend(y_pred.cpu().numpy())
        if loss is not None:
            self.losses.append(loss)

    def compute(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        # Macro F1 is critical for imbalanced classes
        macro_f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        mean_loss = np.mean(self.losses) if self.losses else 0.0
        
        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'loss': mean_loss
        }
