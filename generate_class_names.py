"""
AgroVision AI — Phase 0: Generate class_names.json from dataset
================================================================
Scans the dataset directory and generates the canonical class name list.
This ensures the JSON matches exactly what the model was trained on.

Usage:
    python generate_class_names.py --dataset_dir data/raw
    python generate_class_names.py --dataset_dir "path/to/PlantVillage-Dataset/raw/color"
"""

import os
import sys
import json
import argparse


def generate_class_names(dataset_dir, output_path="shared/class_names.json"):
    """Generate class_names.json from dataset directory structure."""
    if not os.path.isdir(dataset_dir):
        print(f"[✗] Dataset directory not found: {dataset_dir}")
        print(f"    Please provide the correct path to your PlantVillage dataset.")
        sys.exit(1)

    # Get sorted class names (same logic as PlantVillageDataset.__init__)
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if len(classes) == 0:
        print(f"[✗] No subdirectories found in {dataset_dir}")
        sys.exit(1)

    print(f"[✓] Found {len(classes)} classes in: {dataset_dir}")
    for i, cls in enumerate(classes):
        n_images = len([
            f for f in os.listdir(os.path.join(dataset_dir, cls))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"    [{i:2d}] {cls} ({n_images} images)")

    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Saved {len(classes)} class names to: {output_path}")

    # Verify against existing file
    existing_path = output_path
    if os.path.exists(existing_path):
        with open(existing_path, 'r') as f:
            existing = json.load(f)
        if existing == classes:
            print(f"[✓] Verified: matches existing {existing_path}")
        else:
            print(f"[⚠] WARNING: differs from existing {existing_path}!")
            print(f"    Existing has {len(existing)} classes, generated has {len(classes)}")

    return classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/raw",
                        help="Path to dataset root containing class subdirectories")
    parser.add_argument("--output", type=str, default="shared/class_names.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    generate_class_names(args.dataset_dir, args.output)
