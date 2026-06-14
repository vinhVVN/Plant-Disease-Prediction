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

# 3. Preprocessing
# SỬA LẠI: Dùng Resize((224, 224)) (Squash) thay vì CenterCrop để khớp 100% 
# với quá trình resize trên Canvas (JS) và trong Notebook của bạn.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes: bytes, model_name: str = 'efficientnetb0'):
    """
    Hàm này sẽ nhận bytes ảnh, tiền xử lý, chạy inference và trả về kết quả Top-5.
    (Sẽ implement chi tiết cùng với Endpoint /api/predict)
    """
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported")
        
    model = models_dict[model_name]['model']
    
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        results.append({
            "class": CLASS_NAMES[top5_catid[i].item()],
            "confidence": top5_prob[i].item(),
            "class_idx": top5_catid[i].item()
        })
        
    return results, input_tensor, image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(image_tensor: torch.Tensor, target_class: int, model_name: str = 'efficientnetb0'):
    """
    Sinh Grad-CAM heatmap an toàn, đồng bộ không gian tuyệt đối bằng kỹ thuật Un-normalize Tensor.
    """
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported")
        
    model = models_dict[model_name]['model']
    
    # TRẢ VỀ ĐÚNG LAYER BẠN ĐÃ ĐỊNH NGHĨA
    target_layer = models_dict[model_name]['target_layer'] 
    
    # 1. Tính toán Heatmap
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    
    # Bật gradient tạm thời cho Captum (rất quan trọng trong API)
    with torch.set_grad_enabled(True):
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_()
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Shape: [224, 224]
    
    # 2. Đảo ngược chuẩn hóa (Un-normalize) chính xác như Notebook của bạn
    # Squeeze() để loại bỏ batch_size, chuyển từ [1, 3, 224, 224] thành [3, 224, 224]
    img_unnorm = image_tensor.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_unnorm = std * img_unnorm + mean
    img_unnorm = np.clip(img_unnorm, 0, 1) # Ảnh nền bây giờ đã khớp hoàn hảo 100%
    
    # 3. Phủ Heatmap lên ảnh nền (Overlay)
    visualization = show_cam_on_image(img_unnorm, grayscale_cam, use_rgb=True)
    
    # 4. Tạo ảnh Heatmap màu độc lập
    heatmap_color = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 5. Mã hóa Base64 cho API
    def _encode_b64(img_np):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
        
    heatmap_b64 = _encode_b64(heatmap_color)
    overlay_b64 = _encode_b64(visualization)
    
    return heatmap_b64, overlay_b64
