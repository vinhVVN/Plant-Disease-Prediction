import sys
import os
import json
import base64
from io import BytesIO
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from captum.attr import LayerGradCam
import torch.nn.functional as F

# Đảm bảo có thể import từ src (PyTorch Monorepo)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from src.models.factory import create_model

# 1. Load Configurations & Labels
CLASS_NAMES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../shared/class_names.json'))
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# 2. Init Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EfficientNetB0
eff_config = {
    'model': {
        'name': 'efficientnet_b0',
        'num_classes': 38,
        'pretrained': False,
        'dropout_rate': 0.2
    }
}
eff_model = create_model(eff_config)
eff_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../checkpoints/effcientnetb0/best_model_dropout_0.2.pth'))
if os.path.exists(eff_checkpoint):
    state_dict = torch.load(eff_checkpoint, map_location=device)
    # Tệp tạ của EfficientNetB0 bị thiếu prefix 'model.' (Do được train bằng torchvision trực tiếp)
    new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    eff_model.load_state_dict(new_state_dict)
eff_model.to(device)
eff_model.eval()

# MobileNetV3
mb_config = {
    'model': {
        'name': 'mobilenet_v3_small',
        'num_classes': 38,
        'pretrained': False
    }
}
mb_model = create_model(mb_config)
mb_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../checkpoints/mobilenetv3/best_model.pth'))
if os.path.exists(mb_checkpoint):
    mb_model.load_state_dict(torch.load(mb_checkpoint, map_location=device))
mb_model.to(device)
mb_model.eval()

models_dict = {
    'efficientnetb0': {
        'model': eff_model,
        'target_layer': eff_model.model.features[-1] 
    },
    'mobilenetv3': {
        'model': mb_model,
        # CHÚ Ý: KHÔNG DÙNG conv2! conv2 là Pointwise Conv 1x1, không có Spatial Info.
        # Đó là lý do heatmap bị ném ra ngoài background (ngoài vùng lá).
        # BẮT BUỘC phải dùng bnecks[-1] (chứa Depthwise Conv 5x5) để lấy không gian lá.
        'target_layer': mb_model.bnecks[-1]
    }
}

tta_transforms = [
    transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.functional.hflip, transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
]

def _tensor_to_b64(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_unnorm = tensor.cpu().numpy().transpose((1, 2, 0))
    img_unnorm = std * img_unnorm + mean
    img_unnorm = np.clip(img_unnorm, 0, 1)
    img_unnorm_uint8 = np.uint8(img_unnorm * 255)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_unnorm_uint8, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def predict_image(image_bytes: bytes, model_name: str = 'efficientnetb0'):
    """
    Hàm sử dụng TTA (Test-Time Augmentation): tạo batch 4 ảnh, dự đoán, và trả về mean probabilities.
    """
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported")
        
    model = models_dict[model_name]['model']
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Tạo 4 tensors từ 4 transform
    tensors = [t(image) for t in tta_transforms]
    batch_tensor = torch.stack(tensors).to(device) # Shape: (4, 3, 224, 224)
    
    with torch.no_grad():
        output = model(batch_tensor) # Forward pass 1 lần duy nhất cho cả batch 4
        probabilities = torch.nn.functional.softmax(output, dim=1)
        mean_probabilities = probabilities.mean(dim=0) # Trung bình cộng của 4 luồng nhìn
        
    top5_prob, top5_catid = torch.topk(mean_probabilities, 5)
    
    results = []
    for i in range(5):
        results.append({
            "class": CLASS_NAMES[top5_catid[i].item()],
            "confidence": top5_prob[i].item(),
            "class_idx": top5_catid[i].item()
        })
        
    tta_b64_list = [_tensor_to_b64(t) for t in tensors]
    main_tensor = tensors[1].unsqueeze(0).to(device) # Lấy CenterCrop làm đại diện
        
    return results, tta_b64_list, main_tensor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_advanced_xai(image_tensor: torch.Tensor, target_class: int, model_name: str = 'efficientnetb0'):
    """
    Sinh Grad-CAM heatmap, trích xuất Leaf Mask, và tính toán Mức độ nghiêm trọng (Severity Estimation).
    """
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported")
        
    model = models_dict[model_name]['model']
    target_layer = models_dict[model_name]['target_layer'] 
    
    # 1. Tính toán Heatmap bằng Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    
    with torch.set_grad_enabled(True):
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_()
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Shape: [224, 224]
        
        # Trích xuất Feature Map (Layer 1 - Conv1)
        try:
            if model_name == 'mobilenetv3':
                first_layer_out = model.features[0][0](image_tensor)
            else:
                first_layer_out = model.features[0][0](image_tensor)
                
            # Lấy Kênh 0 (Filter 0), áp dụng ReLU
            fmap = F.relu(first_layer_out[0, 0]).cpu().detach().numpy()
            
            # Chuẩn hóa Min-Max thủ công [0, 255]
            fmap_min, fmap_max = fmap.min(), fmap.max()
            if fmap_max > fmap_min:
                fmap_norm = (fmap - fmap_min) / (fmap_max - fmap_min) * 255.0
            else:
                fmap_norm = np.zeros_like(fmap)
                
            fmap_norm = np.uint8(fmap_norm)
            fmap_color = cv2.applyColorMap(fmap_norm, cv2.COLORMAP_VIRIDIS)
            fmap_color = cv2.cvtColor(fmap_color, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("Feature Map Extraction Error:", e)
            fmap_color = np.zeros((224, 224, 3), dtype=np.uint8)
            
    # 2. Đảo ngược chuẩn hóa (Pipeline Trace)
    img_unnorm = image_tensor.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_unnorm = std * img_unnorm + mean
    img_unnorm = np.clip(img_unnorm, 0, 1) 
    img_unnorm_uint8 = np.uint8(img_unnorm * 255)
    
    # 3. Tính toán Severity (Leaf Mask & Thresholding)
    hsv = cv2.cvtColor(img_unnorm_uint8, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    _, leaf_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tạo ảnh Leaf Mask (Trắng đen) để hiển thị lên Pipeline Trace
    leaf_mask_visual = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2RGB)
    
    leaf_area = np.sum(leaf_mask > 0)
    # Lấy vùng heatmap mạnh (>0.6) giao với lá
    disease_mask = grayscale_cam > 0.6
    disease_area = np.sum((disease_mask) & (leaf_mask > 0))
    
    severity_percentage = 0.0
    severity_level = "Healthy"
    if leaf_area > 0:
        severity_percentage = (disease_area / leaf_area) * 100
        if severity_percentage < 5:
            severity_level = "Mild"
        elif severity_percentage < 20:
            severity_level = "Moderate"
        else:
            severity_level = "Severe"
            
    # Nếu là lá khỏe mạnh thì severity = 0
    if "healthy" in CLASS_NAMES[target_class].lower():
        severity_percentage = 0.0
        severity_level = "None"
    
    # 4. Phủ Heatmap lên ảnh nền (Overlay)
    visualization = show_cam_on_image(img_unnorm, grayscale_cam, use_rgb=True)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 5. Mã hóa Base64 cho API
    def _encode_b64(img_np):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
        
    return {
        "pipeline_trace_b64": _encode_b64(img_unnorm_uint8),
        "feature_map_b64": _encode_b64(fmap_color),
        "leaf_mask_b64": _encode_b64(leaf_mask_visual),
        "heatmap_b64": _encode_b64(heatmap_color),
        "overlay_b64": _encode_b64(visualization),
        "severity": {
            "percentage": round(severity_percentage, 2),
            "level": severity_level
        }
    }
