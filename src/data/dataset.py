import os
from PIL import Image
from torch.utils.data import Dataset

class PlantVillageDataset(Dataset):
    """
    Custom PyTorch Dataset for the PlantVillage dataset.
    Expects a directory structure where subdirectories are class names:
    dataset_dir/
    ├── Apple___Apple_scab/
    │   ├── image1.jpg
    │   └── ...
    ├── Apple___Black_rot/
    │   └── ...
    └── ...
    """
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(dataset_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
