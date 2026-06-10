from torchvision import transforms

def get_baseline_transforms(image_size=224):
    """
    Returns the strict baseline transforms (No complex data augmentation).
    Matches the Khan 2023 paper methodology for the PlantVillage dataset.
    """
    baseline_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return baseline_transform

def get_hpo_transforms(image_size=224):
    """
    Returns an expanded search space configuration for Data Augmentation,
    to be used EXCLUSIVELY during Optuna Hyperparameter Optimization trials.
    """
    hpo_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return hpo_transform
