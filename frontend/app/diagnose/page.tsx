"use client";

import { useState, useRef } from "react";
import { Upload, ImageIcon, Leaf, Cpu, Loader2, ArrowRight, Cloud, ShieldAlert, AlertCircle, Microscope } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runEdgeInference, InferenceResult } from "@/lib/onnx-inference";
import { compressImage, savePrediction } from "@/lib/db";
import { runCloudInference, CloudPredictionResult } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function DiagnosePage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  
  // State quản lý mô hình
  const [model, setModel] = useState<'mobilenetv3' | 'efficientnetb0'>('mobilenetv3');

  // Kết quả
  const [edgeResults, setEdgeResults] = useState<InferenceResult[] | null>(null);
  const [cloudResult, setCloudResult] = useState<CloudPredictionResult | null>(null);
  
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [pipelineStep, setPipelineStep] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setEdgeResults(null);
      setCloudResult(null);
      setPipelineStep(0);
    }
  };

  const startDiagnosis = async () => {
    if (!image) return;
    
    setIsInferring(true);
    setPipelineStep(1); // Bước 1
    setEdgeResults(null);
    setCloudResult(null);
    
    try {
      const startTime = performance.now();

      if (model === 'mobilenetv3') {
        // --- EDGE INFERENCE CHO MOBILENET ---
        await new Promise(r => setTimeout(r, 600));
        setPipelineStep(2); // CNN Inference
        
        // Gọi API lên Cloud ĐỂ LẤY GRAD-CAM CHẠY SONG SONG
        const cloudTask = runCloudInference(image, 'mobilenetv3');
        
        await new Promise(r => setTimeout(r, 600));
        setPipelineStep(3); // Softmax
        
        const predictions = await runEdgeInference(image);
        const endTime = performance.now();
        const timeMs = Math.round(endTime - startTime);
        
        setEdgeResults(predictions);
        setInferenceTime(timeMs);
        setPipelineStep(4);
        
        // Chờ Grad-CAM về để show ở dưới
        try {
          const cResult = await cloudTask;
          setCloudResult(cResult);
        } catch (e) {
          console.error("Lỗi lấy Grad-CAM cho MobileNetV3", e);
        }

        // Lưu DB
        compressImage(image).then(compressed => {
          compressImage(image, 100, 0.5).then(thumbnail => {
            savePrediction({
              imageBlob: compressed,
              imageThumbnail: thumbnail,
              modelUsed: 'mobilenetv3',
              predictedClass: predictions[0].className,
              confidence: predictions[0].confidence,
              inferenceTimeMs: timeMs
            });
          });
        });

      } else {
        // --- CLOUD INFERENCE CHO EFFICIENTNET ---
        await new Promise(r => setTimeout(r, 600));
        setPipelineStep(2); // Cloud Inference
        
        const data = await runCloudInference(image, 'efficientnetb0');
        
        setPipelineStep(3); // Postprocess
        await new Promise(r => setTimeout(r, 400));
        
        const endTime = performance.now();
        const timeMs = Math.round(endTime - startTime);
        
        setCloudResult(data);
        setInferenceTime(data.inference_time_ms); // Lấy thời gian thực của backend
        setPipelineStep(4);
        
        // Lưu DB
        compressImage(image).then(compressed => {
          compressImage(image, 100, 0.5).then(thumbnail => {
            savePrediction({
              imageBlob: compressed,
              imageThumbnail: thumbnail,
              modelUsed: 'efficientnetb0',
              predictedClass: data.predicted_class,
              confidence: data.confidence,
              inferenceTimeMs: data.inference_time_ms
            });
          });
        });
      }

    } catch (error) {
      console.error("Lỗi chẩn đoán:", error);
      alert("Lỗi quá trình chẩn đoán! Hãy kiểm tra console.");
    } finally {
      setIsInferring(false);
    }
  };

  const isMobileNet = model === 'mobilenetv3';

  // Format data cho Recharts nếu có
  const cloudChartData = cloudResult?.top5.map(item => ({
    name: item.class.replace(/___/g, ' - ').replace(/_/g, ' '),
    probability: Number((item.prob * 100).toFixed(1))
  }));

  // Chuẩn bị dữ liệu Top 5 (lấy từ Edge nếu là MobileNet, lấy từ Cloud nếu là EfficientNet)
  const displayResults = isMobileNet 
    ? edgeResults?.map(r => ({ className: r.className, confidence: r.confidence }))
    : cloudResult?.top5.map(r => ({ className: r.class, confidence: r.prob }));

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Header & Mode Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-2">
            <Microscope className="text-blue-500 w-8 h-8" />
            AI Diagnosis Center
          </h1>
          <p className="text-slate-500 mt-2">
            Powered by Edge AI & Cloud XAI
          </p>
        </div>
        
        {/* Toggle Model */}
        <div className="flex bg-slate-200 rounded-xl p-1 w-fit">
          <button 
            onClick={() => {
              setModel('mobilenetv3');
              if(image) { setEdgeResults(null); setCloudResult(null); setPipelineStep(0); }
            }}
            className={`px-6 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-all ${isMobileNet ? 'bg-white shadow-sm text-green-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <Cpu className="w-4 h-4" /> MobileNetV3
          </button>
          <button 
            onClick={() => {
              setModel('efficientnetb0');
              if(image) { setEdgeResults(null); setCloudResult(null); setPipelineStep(0); }
            }}
            className={`px-6 py-2 rounded-lg font-medium text-sm flex items-center gap-2 transition-all ${!isMobileNet ? 'bg-white shadow-sm text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <Cloud className="w-4 h-4" /> EfficientNetB0
          </button>
        </div>
      </div>

      {/* TOP SECTION: Upload & Pipeline / Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* PANEL 1: Upload */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
          <div className="mb-4 font-semibold text-lg flex items-center justify-between">
            <span>Input Image</span>
            {isMobileNet ? (
              <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">Edge Mode</span>
            ) : (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">Cloud Mode</span>
            )}
          </div>
          
          <input 
            type="file" 
            accept="image/*" 
            className="hidden" 
            ref={fileInputRef}
            onChange={handleImageUpload}
          />
          
          {!preview ? (
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed border-slate-300 rounded-xl h-80 flex flex-col items-center justify-center cursor-pointer transition-colors ${isMobileNet ? 'hover:border-green-400 hover:bg-green-50' : 'hover:border-blue-400 hover:bg-blue-50'}`}
            >
              <Upload className="w-12 h-12 text-slate-400 mb-4" />
              <p className="text-slate-600 font-medium">Click to upload plant leaf</p>
            </div>
          ) : (
            <div className="relative h-80 rounded-xl overflow-hidden bg-black flex items-center justify-center">
              <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain" />
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="absolute top-4 right-4 bg-white/90 backdrop-blur text-sm px-4 py-2 rounded-lg font-medium shadow hover:bg-white"
              >
                Change Image
              </button>
            </div>
          )}

          <button
            onClick={startDiagnosis}
            disabled={!image || isInferring}
            className={`w-full mt-6 text-white font-bold py-4 rounded-xl shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all ${
              isMobileNet ? 'bg-green-600 shadow-green-600/20 hover:bg-green-700' : 'bg-blue-600 shadow-blue-600/20 hover:bg-blue-700'
            }`}
          >
            {isInferring ? (
              <><Loader2 className="animate-spin" /> Processing...</>
            ) : (
              <><Leaf /> Diagnose Plant Disease</>
            )}
          </button>
        </div>

        {/* PANEL 2: Pipeline & Results Card */}
        <div className="space-y-6">
          {/* Pipeline Animation */}
          <div className="bg-slate-900 rounded-2xl p-6 shadow-sm text-white overflow-hidden relative">
            <div className="font-semibold text-lg text-slate-300 mb-6">
              {isMobileNet ? 'ONNX Edge Pipeline' : 'FastAPI Cloud Pipeline'}
            </div>
            <div className="flex items-center justify-between text-sm font-medium">
              {[
                { step: 1, label: isMobileNet ? "Canvas Resize" : "Upload Image", icon: <ImageIcon className="w-5 h-5 text-white" /> },
                { step: 2, label: isMobileNet ? "WASM Backend" : "Cloud Inference", icon: isMobileNet ? <Cpu className="w-5 h-5 text-white" /> : <Cloud className="w-5 h-5 text-white" /> },
                { step: 3, label: "Top-5 & Grad-CAM", icon: <ArrowRight className="w-5 h-5 text-white" /> }
              ].map((item, idx) => (
                <div key={idx} className="flex flex-col items-center relative z-10 w-1/3 text-center">
                  <motion.div 
                    animate={{
                      backgroundColor: pipelineStep >= item.step ? (isMobileNet ? "#22c55e" : "#3b82f6") : "#334155",
                      scale: pipelineStep === item.step ? 1.1 : 1,
                    }}
                    className="w-12 h-12 rounded-full flex items-center justify-center mb-2 shadow-lg"
                  >
                    {item.icon}
                  </motion.div>
                  <div className={pipelineStep >= item.step ? (isMobileNet ? "text-green-400" : "text-blue-400") : "text-slate-500"}>
                    {item.label}
                  </div>
                </div>
              ))}
            </div>
            
            {/* Thanh kết nối */}
            <div className="absolute top-12 left-16 right-16 h-1 bg-slate-800 -z-0">
              <motion.div 
                className={`h-full ${isMobileNet ? 'bg-green-500' : 'bg-blue-500'}`}
                initial={{ width: "0%" }}
                animate={{ width: pipelineStep === 0 ? "0%" : pipelineStep === 1 ? "25%" : pipelineStep === 2 ? "75%" : "100%" }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>

          {/* Results Card */}
          <AnimatePresence>
            {displayResults && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100"
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="font-semibold text-lg">Top Predictions</h3>
                  <div className="text-sm text-slate-500 bg-slate-100 px-3 py-1 rounded-full">
                    {inferenceTime} ms
                  </div>
                </div>

                <div className="space-y-4">
                  {displayResults.map((res, idx) => (
                    <div key={idx}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-slate-700 truncate max-w-[70%]">
                          {res.className.replace(/___/g, ' - ').replace(/_/g, ' ')}
                        </span>
                        <span className={`font-bold ${isMobileNet ? 'text-green-600' : 'text-blue-600'}`}>
                          {(res.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${res.confidence * 100}%` }}
                          transition={{ duration: 1, delay: idx * 0.1 }}
                          className={`h-full rounded-full ${isMobileNet ? (idx === 0 ? 'bg-green-500' : 'bg-green-300') : (idx === 0 ? 'bg-blue-500' : 'bg-blue-300')}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* BOTTOM SECTION: Explainable AI (Grad-CAM) */}
      <AnimatePresence>
        {cloudResult && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8 pt-4 border-t border-slate-200"
          >
            <div className="text-center">
              <h2 className="text-2xl font-bold text-slate-800">Explainable AI (XAI) Analysis</h2>
              <p className="text-slate-500">Grad-CAM visualization generated from backend</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Original Image */}
              <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-100 flex flex-col items-center gap-3">
                <div className="w-full aspect-square bg-black rounded-xl overflow-hidden">
                  <img src={preview!} alt="Original" className="w-full h-full object-contain" />
                </div>
                <span className="font-medium text-slate-700">Original Scan</span>
              </div>
              
              {/* Heatmap */}
              <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-100 flex flex-col items-center gap-3">
                <div className="w-full aspect-square bg-black rounded-xl overflow-hidden relative">
                  <img src={cloudResult.gradcam_heatmap} alt="Heatmap" className="w-full h-full object-contain" />
                </div>
                <span className="font-medium text-slate-700">Activation Heatmap</span>
              </div>

              {/* Overlay + Recommendation */}
              <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-100 flex flex-col gap-3">
                <div className="w-full aspect-square bg-black rounded-xl overflow-hidden relative border-2 border-red-400">
                  <img src={cloudResult.gradcam_overlay} alt="Overlay" className="w-full h-full object-contain" />
                  <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded shadow font-bold">
                    Focus Area
                  </div>
                </div>
                
                <div className={`mt-2 rounded-xl p-3 border text-sm ${
                  cloudResult.recommendation.priority === 'high' ? 'bg-red-50 border-red-100 text-red-800' : 
                  cloudResult.recommendation.priority === 'medium' ? 'bg-orange-50 border-orange-100 text-orange-800' : 
                  'bg-emerald-50 border-emerald-100 text-emerald-800'
                }`}>
                  <div className="font-bold flex items-center gap-1 mb-1">
                    <ShieldAlert className="w-4 h-4" /> Recommendation
                  </div>
                  <div className="font-medium">{cloudResult.recommendation.action}</div>
                </div>
              </div>
            </div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
