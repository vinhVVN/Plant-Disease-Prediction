import sys
import os
# Đảm bảo có thể import từ src (PyTorch Monorepo)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.services.inference import predict_image, generate_gradcam

from backend.app.routers.predict import router as predict_router

app = FastAPI(title="AgroVision AI - Cloud Backend")

# Cho phép origin từ Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI is running"}

@app.get("/")
async def root():
    return {"message": "Welcome to AgroVision API"}
