from src.models.efficientnet import EfficientNetB0
from src.models.mobilenet_v3 import MobileNetV3Small

def create_model(config):
    """
    Factory function to create models based on the YAML configuration.
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name')
    num_classes = model_config.get('num_classes', 38)
    pretrained = model_config.get('pretrained', True)
    
    if model_name == 'efficientnet_b0':
        dropout_rate = model_config.get('dropout_rate', 0.2)
        model = EfficientNetB0(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif model_name == 'mobilenet_v3_small':
        model = MobileNetV3Small(
            num_classes=num_classes
        )
        # Note: Pretrained weights for custom MobileNetV3 architectures
        # are typically loaded via a separate weight_loader script,
        # but the instantiation is handled here.
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model
