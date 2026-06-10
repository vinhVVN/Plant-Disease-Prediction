import os
import yaml
import torch
from src.models.mobilenet_v3 import MobileNetV3Small

def export_model_for_edge(config_path="configs/default_config.yaml"):
    """
    Exports the trained PyTorch model to TorchScript, ONNX (Opset 17), 
    and applies Dynamic Quantization for Edge Devices.
    """
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    num_classes = config['model']['num_classes']
    save_dir = config['training']['save_dir']
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found. Ensure the model has been trained.")
        return

    # 2. Instantiate Architecture and Load Trained Weights
    model = MobileNetV3Small(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Successfully loaded trained weights from {checkpoint_path}.")
    
    # Calculate Original Size
    orig_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Original Model Size: {orig_size_mb:.2f} MB\n")
    
    # 3. Create a Dummy Input Tensor (Batch Size 1, 3 Channels, 224x224 Height/Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # --- Action 1: Export to TorchScript ---
    ts_path = os.path.join(save_dir, "best_model.pt")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        print(f"Action 1 Success: TorchScript model saved to {ts_path}")
    except Exception as e:
        print(f"Action 1 Failed (TorchScript Export): {e}")
        
    # --- Action 2: Export to ONNX (Opset 17) ---
    onnx_path = os.path.join(save_dir, "best_model.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Action 2 Success: ONNX model (Opset 17) saved to {onnx_path}")
    except Exception as e:
        print(f"Action 2 Failed (ONNX Export): {e}")

    # --- Action 3: Dynamic Quantization ---
    quantized_path = os.path.join(save_dir, "best_model_quantized.pt")
    try:
        # Perform dynamic quantization specifically targeting nn.Linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Save quantized model using TorchScript for edge deployment
        traced_quantized_model = torch.jit.trace(quantized_model, dummy_input)
        traced_quantized_model.save(quantized_path)
        
        quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        
        print(f"Action 3 Success: Dynamic Quantization applied to Linear layers.")
        print(f"Quantized Model saved to {quantized_path}")
        print(f"\n Kết quả nén")
        print(f"Original Size : {orig_size_mb:.2f} MB")
        print(f"Quantized Size: {quant_size_mb:.2f} MB")
        print(f"Compression   : {orig_size_mb / quant_size_mb:.2f}x smaller")
        
    except Exception as e:
        print(f"Action 3 Failed (Dynamic Quantization): {e}")

if __name__ == '__main__':
    export_model_for_edge()
