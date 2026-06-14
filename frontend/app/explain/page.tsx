"use client";

import { useState, useRef } from "react";
import { Upload, Microscope, Loader2, AlertCircle, ShieldAlert } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runCloudInference, CloudPredictionResult } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function XAICenterPage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  const [result, setResult] = useState<CloudPredictionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const startAnalysis = async () => {
    if (!image) return;
    setIsInferring(true);
    
    try {
      const data = await runCloudInference(image);
      setResult(data);
    } catch (error) {
      console.error("Lỗi chẩn đoán Cloud:", error);
      alert("Có lỗi xảy ra khi gọi FastAPI Backend! Hãy đảm bảo Backend đang chạy.");
    } finally {
      setIsInferring(false);
    }
  };

  // Chuẩn bị dữ liệu cho biểu đồ Recharts
  const chartData = result?.top5.map(item => ({
    name: item.class.replace(/___/g, ' - ').replace(/_/g, ' '),
    probability: Number((item.prob * 100).toFixed(1))
  }));

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-2">
            <Microscope className="text-blue-500 w-8 h-8" />
            Explainable AI Center
          </h1>
          <p className="text-slate-500 mt-2">
            Powered by EfficientNetB0 & Captum Grad-CAM on Cloud
          </p>
        </div>
        <div className="bg-blue-100 text-blue-700 px-4 py-2 rounded-full font-medium text-sm flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
          Cloud Mode Active
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Upload Panel (1 cột) */}
        <div className="lg:col-span-1 bg-white rounded-2xl p-6 shadow-sm border border-slate-100 h-fit">
          <div className="mb-4 font-semibold text-lg">Input Image</div>
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
              className="border-2 border-dashed border-slate-300 rounded-xl h-64 flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
            >
              <Upload className="w-10 h-10 text-slate-400 mb-4" />
              <p className="text-slate-600 font-medium">Upload for Deep Analysis</p>
            </div>
          ) : (
            <div className="relative h-64 rounded-xl overflow-hidden bg-black flex items-center justify-center">
              <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain" />
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="absolute top-4 right-4 bg-white/90 backdrop-blur text-sm px-4 py-2 rounded-lg font-medium shadow hover:bg-white"
              >
                Change
              </button>
            </div>
          )}

          <button
            onClick={startAnalysis}
            disabled={!image || isInferring}
            className="w-full mt-6 bg-blue-600 text-white font-bold py-4 rounded-xl shadow-lg shadow-blue-600/20 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
          >
            {isInferring ? (
              <><Loader2 className="animate-spin" /> Analyzing features...</>
            ) : (
              <><Microscope /> Extract Heatmap</>
            )}
          </button>
        </div>

        {/* XAI Visualization Panels (2 cột) */}
        <div className="lg:col-span-2">
          {result ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100"
            >
              <h3 className="font-semibold text-lg mb-4 flex justify-between items-center">
                <span>Grad-CAM Feature Activation</span>
                <span className="text-xs font-normal bg-slate-100 px-3 py-1 rounded-full text-slate-500">
                  Target Layer: features[-1]
                </span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Original Image Panel */}
                <div className="flex flex-col items-center gap-2">
                  <div className="w-full aspect-square bg-black rounded-xl overflow-hidden">
                    <img src={preview!} alt="Original" className="w-full h-full object-contain" />
                  </div>
                  <span className="text-sm font-medium text-slate-600">Original Image</span>
                </div>
                
                {/* Heatmap Panel */}
                <div className="flex flex-col items-center gap-2">
                  <div className="w-full aspect-square bg-black rounded-xl overflow-hidden">
                    <img src={result.gradcam_heatmap} alt="Heatmap" className="w-full h-full object-contain" />
                  </div>
                  <span className="text-sm font-medium text-slate-600">Pure Heatmap</span>
                </div>

                {/* Overlay Panel */}
                <div className="flex flex-col items-center gap-2">
                  <div className="w-full aspect-square bg-black rounded-xl overflow-hidden relative border-2 border-blue-400">
                    <img src={result.gradcam_overlay} alt="Overlay" className="w-full h-full object-contain" />
                    <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded shadow-lg font-bold">
                      Focus Area
                    </div>
                  </div>
                  <span className="text-sm font-medium text-blue-600">Overlay Analysis</span>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-2xl h-full min-h-[300px] flex items-center justify-center">
              <p className="text-slate-400">Upload an image and run analysis to see Grad-CAM results</p>
            </div>
          )}
        </div>
      </div>

      {/* Kết quả & Recommendation */}
      <AnimatePresence>
        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          >
            {/* Chart & Explain Box */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
              <h3 className="font-semibold text-lg mb-6">Confidence Distribution</h3>
              <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} hide />
                    <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="probability" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 flex gap-3">
                <AlertCircle className="text-blue-500 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-blue-900 mb-1">Why this prediction?</h4>
                  <p className="text-sm text-blue-800">
                    The model focused on the <span className="font-bold">red-highlighted regions</span> in the overlay image. 
                    These specific texture patterns strongly correlate with the visual characteristics of 
                    <span className="font-bold"> {result.predicted_class.replace(/___/g, ' - ').replace(/_/g, ' ')} </span> 
                    learned during training.
                  </p>
                </div>
              </div>
            </div>

            {/* Recommendation Box */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
              <h3 className="font-semibold text-lg mb-6">Diagnosis & Recommendation</h3>
              
              <div className="mb-8">
                <div className="text-sm text-slate-500 uppercase tracking-wider mb-1">Primary Diagnosis</div>
                <div className="text-2xl font-bold text-slate-800">
                  {result.predicted_class.replace(/___/g, ' - ').replace(/_/g, ' ')}
                </div>
                <div className="inline-flex items-center gap-1 mt-2 bg-green-100 text-green-700 px-3 py-1 rounded-full font-medium text-sm">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
                <div className="inline-flex items-center gap-1 mt-2 ml-2 bg-slate-100 text-slate-700 px-3 py-1 rounded-full font-medium text-sm">
                  Server Time: {result.inference_time_ms} ms
                </div>
              </div>

              <div className={`rounded-xl p-5 border ${
                result.recommendation.priority === 'high' ? 'bg-red-50 border-red-100' : 
                result.recommendation.priority === 'medium' ? 'bg-orange-50 border-orange-100' : 
                'bg-emerald-50 border-emerald-100'
              }`}>
                <div className="flex items-start gap-3 mb-3">
                  <ShieldAlert className={`shrink-0 w-6 h-6 ${
                    result.recommendation.priority === 'high' ? 'text-red-500' : 
                    result.recommendation.priority === 'medium' ? 'text-orange-500' : 
                    'text-emerald-500'
                  }`} />
                  <div>
                    <h4 className="font-bold text-slate-800 text-lg">Action Plan</h4>
                    <p className="text-sm text-slate-600 mt-1">{result.recommendation.description}</p>
                  </div>
                </div>
                
                <div className="mt-4 bg-white/60 rounded-lg p-4 font-medium text-slate-800 border border-white/40 shadow-sm">
                  👉 {result.recommendation.action}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
