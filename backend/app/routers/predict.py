import time
import json
import os
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.inference import predict_image, generate_gradcam

router = APIRouter()

# Đọc file recommendations.json
RECOMMENDATIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../shared/recommendations.json'))
if os.path.exists(RECOMMENDATIONS_PATH):
    with open(RECOMMENDATIONS_PATH, 'r', encoding='utf-8') as f:
        RECOMMENDATIONS = json.load(f)
else:
    RECOMMENDATIONS = {}

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

# ... (omitted line so start from 18)
@router.post("/api/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnetb0")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    image_bytes = await file.read()
    
    start_time = time.time()
    
    try:
        # 1. Chạy Forward pass (không tính gradient) để lấy kết quả Top 5
        # Tránh bẫy lưu gradient không cần thiết trong quá trình predict thông thường
        with torch.no_grad():
            top5_results, input_tensor, original_image = predict_image(image_bytes, model_name=model_name)
            
        target_class_idx = top5_results[0]['class_idx']
        predicted_class_name = top5_results[0]['class']
        confidence = top5_results[0]['confidence']
        
        # 2. Sinh Grad-CAM (Bắt buộc PHẢI bật gradient)
        # Captum yêu cầu tính backward pass (gradient) từ output ngược về input
        with torch.set_grad_enabled(True):
            # Input tensor cần require_grad để tính được heatmap
            input_tensor.requires_grad_()
            heatmap_b64, overlay_b64 = generate_gradcam(input_tensor, target_class_idx, model_name=model_name)
            
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # 3. Lấy recommendation dựa trên tên bệnh
        recommendation = RECOMMENDATIONS.get(predicted_class_name, {
            "action": "Không có khuyến nghị cụ thể.",
            "priority": "low",
            "description": "Vui lòng theo dõi thêm tình trạng của cây."
        })
        
        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "top5": [{"class": res["class"], "prob": res["confidence"]} for res in top5_results],
            "gradcam_heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
            "gradcam_overlay": f"data:image/jpeg;base64,{overlay_b64}",
            "inference_time_ms": inference_time_ms,
            "recommendation": recommendation
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
