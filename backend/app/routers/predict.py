import time
import json
import os
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.app.services.inference import predict_image, generate_advanced_xai

router = APIRouter()

# Đọc file recommendations.json
RECOMMENDATIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../shared/recommendations.json'))
if os.path.exists(RECOMMENDATIONS_PATH):
    with open(RECOMMENDATIONS_PATH, 'r', encoding='utf-8') as f:
        RECOMMENDATIONS = json.load(f)
else:
    RECOMMENDATIONS = {}

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
        with torch.no_grad():
            top5_results, input_tensor, original_image = predict_image(image_bytes, model_name=model_name)
            
        target_class_idx = top5_results[0]['class_idx']
        predicted_class_name = top5_results[0]['class']
        confidence = top5_results[0]['confidence']
        
        # Entropy / Uncertainty Analysis
        uncertainty_warning = False
        if len(top5_results) >= 2:
            gap = top5_results[0]['confidence'] - top5_results[1]['confidence']
            if gap < 0.2: # Nếu chênh lệch < 20% thì cảnh báo không chắc chắn
                uncertainty_warning = True
        
        with torch.set_grad_enabled(True):
            input_tensor.requires_grad_()
            advanced_xai = generate_advanced_xai(input_tensor, target_class_idx, model_name=model_name)
            
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        recommendation = RECOMMENDATIONS.get(predicted_class_name, {
            "action": "Không có khuyến nghị cụ thể.",
            "priority": "low",
            "description": "Vui lòng theo dõi thêm tình trạng của cây."
        })
        
        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "uncertainty_warning": uncertainty_warning,
            "top5": top5_results,
            "inference_time_ms": inference_time_ms,
            "recommendation": recommendation,
            "xai": advanced_xai
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
