import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.models.factory import create_model
from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_transforms
from src.engine.train import Trainer
from src.engine.evaluate import evaluate_model
from src.engine.tune import tune_hyperparameters
from src.engine.export import export_model_for_edge

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Classification Monorepo")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "tune", "export"], help="Execution mode")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == "tune":
        tune_hyperparameters(args.config)
        return
        
    if args.mode == "export":
        export_model_for_edge(args.config)
        return

    # Setup for train or evaluate
    model_name = config['model']['name']
    image_size = config['data']['image_size']
    dataset_dir = config['data']['dataset_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    train_transform = get_transforms(model_name, is_train=True, image_size=image_size)
    eval_transform = get_transforms(model_name, is_train=False, image_size=image_size)

    full_dataset_train = PlantVillageDataset(dataset_dir, transform=train_transform)
    full_dataset_eval = PlantVillageDataset(dataset_dir, transform=eval_transform)
    class_names = full_dataset_train.classes

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset_train), generator=generator).tolist()
    train_size = int(0.8 * len(full_dataset_train))

    if args.mode == "train":
        train_dataset = Subset(full_dataset_train, indices[:train_size])
        val_dataset = Subset(full_dataset_eval, indices[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        model = create_model(config)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()

    elif args.mode == "evaluate":
        val_dataset = Subset(full_dataset_eval, indices[train_size:])
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        model = create_model(config)
        best_model_path = config['training']['save_dir'] + "/best_model.pth"
        model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=True))
        
        evaluate_model(model, test_loader, class_names, config)

if __name__ == "__main__":
    main()
