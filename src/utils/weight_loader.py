import torch
import torchvision.models as models
import sys
import os
from src.models.mobilenet_v3 import MobileNetV3Small

def migrate_imagenet_weights(custom_model, num_classes=38):
    """
    Surgically migrates pre-trained ImageNet weights from torchvision to our custom model.
    Uses strict=True to enforce academic and structural integrity.
    """
    # 1. Load pretrained model
    pretrained_model = models.mobilenet_v3_small(weights='DEFAULT')
    pretrained_state_dict = pretrained_model.state_dict()
    
    # 2. Get our custom model's state_dict
    custom_state_dict = custom_model.state_dict()
    
    # Extract keys
    pretrained_keys = list(pretrained_state_dict.keys())
    custom_keys = list(custom_state_dict.keys())
    
    # Sanity check on length
    if len(pretrained_keys) != len(custom_keys):
        raise ValueError(f"State dict length mismatch! Pretrained has {len(pretrained_keys)} keys, Custom has {len(custom_keys)} keys.")
        
    mapped_dict = {}
    
    # 3. Ordered mapping (exploiting cell-by-cell architectural symmetry)
    for p_key, c_key in zip(pretrained_keys, custom_keys):
        # 4. Handle Classification Head (last layer)
        # Keep our custom initialized weights for the final classification layer
        # because the number of classes differs (1000 vs 38).
        if 'classifier.3.weight' in c_key or 'classifier.3.bias' in c_key:
            mapped_dict[c_key] = custom_state_dict[c_key]
        else:
            mapped_dict[c_key] = pretrained_state_dict[p_key]
            
    # 5. Validation using strict=True
    custom_model.load_state_dict(mapped_dict, strict=True)
    print("State_Dict Surgery Successful: ImageNet weights mapped and loaded with strict=True.")
    
    return custom_model


if __name__ == '__main__':
    # Add project root to sys.path to allow absolute imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    
    # Instantiate custom architecture
    custom_model = MobileNetV3Small(num_classes=38)
    
    # Execute surgery
    try:
        migrated_model = migrate_imagenet_weights(custom_model, num_classes=38)
        print("Model is ready for fine-tuning!")
    except Exception as e:
        print(f"Surgery failed! Exception: {e}")
