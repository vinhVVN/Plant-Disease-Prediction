"use client";

import { useState, useRef } from "react";
import { Upload, ImageIcon, Leaf, Cpu, Loader2, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runEdgeInference, InferenceResult } from "@/lib/onnx-inference";
import { compressImage, savePrediction } from "@/lib/db";

export default function DiagnosePage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isInferring, setIsInferring] = useState(false);
  const [results, setResults] = useState<InferenceResult[] | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [pipelineStep, setPipelineStep] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setPipelineStep(0);
    }
  };

  const startDiagnosis = async () => {
    if (!image) return;
    
    setIsInferring(true);
    setPipelineStep(1); // Resize & Normalize
    
    try {
      const startTime = performance.now();
      
      // Chờ một chút để animation chạy
      await new Promise(r => setTimeout(r, 600));
      setPipelineStep(2); // CNN Inference
      
      await new Promise(r => setTimeout(r, 600));
      setPipelineStep(3); // Softmax
      
      // Thực thi inference thật
      const predictions = await runEdgeInference(image);
      const endTime = performance.now();
      const timeMs = Math.round(endTime - startTime);
      
      setResults(predictions);
      setInferenceTime(timeMs);
      setPipelineStep(4); // Hoàn thành
      
      // Nén ảnh và lưu vào DB (bất đồng bộ, không block UI)
      compressImage(image).then(compressed => {
        compressImage(image, 100, 0.5).then(thumbnail => {
          savePrediction({
            imageBlob: compressed,
            imageThumbnail: thumbnail,
            modelUsed: 'mobilenet_edge',
            predictedClass: predictions[0].className,
            confidence: predictions[0].confidence,
            inferenceTimeMs: timeMs
          });
        });
      });
      
    } catch (error) {
      console.error("Lỗi chẩn đoán:", error);
      alert("Có lỗi xảy ra khi chạy ONNX Inference!");
    } finally {
      setIsInferring(false);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-2">
            <Cpu className="text-green-500 w-8 h-8" />
            AI Diagnosis Center
          </h1>
          <p className="text-slate-500 mt-2">
            100% Client-Side Inference powered by ONNX Runtime Web
          </p>
        </div>
        <div className="bg-green-100 text-green-700 px-4 py-2 rounded-full font-medium text-sm flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          Edge Mode Active
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Khung bên trái: Upload */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
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
              className="border-2 border-dashed border-slate-300 rounded-xl h-80 flex flex-col items-center justify-center cursor-pointer hover:border-green-400 hover:bg-green-50 transition-colors"
            >
              <Upload className="w-12 h-12 text-slate-400 mb-4" />
              <p className="text-slate-600 font-medium">Click to upload plant leaf</p>
              <p className="text-slate-400 text-sm mt-1">JPEG, PNG up to 10MB</p>
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
            className="w-full mt-6 bg-green-600 text-white font-bold py-4 rounded-xl shadow-lg shadow-green-600/20 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
          >
            {isInferring ? (
              <><Loader2 className="animate-spin" /> Processing at Edge...</>
            ) : (
              <><Leaf /> Diagnose Plant Disease</>
            )}
          </button>
        </div>

        {/* Khung bên phải: Kết quả & Pipeline */}
        <div className="space-y-6">
          {/* Pipeline Animation */}
          <div className="bg-slate-900 rounded-2xl p-6 shadow-sm text-white overflow-hidden relative">
            <div className="font-semibold text-lg text-slate-300 mb-6">ONNX Inference Pipeline</div>
            <div className="flex items-center justify-between text-sm font-medium">
              {[
                { step: 1, label: "Preprocess", desc: "Canvas Resize" },
                { step: 2, label: "MobileNetV3", desc: "WASM Backend" },
                { step: 3, label: "Postprocess", desc: "Softmax Top-5" }
              ].map((item, idx) => (
                <div key={idx} className="flex flex-col items-center relative z-10">
                  <motion.div 
                    animate={{
                      backgroundColor: pipelineStep >= item.step ? "#22c55e" : "#334155",
                      scale: pipelineStep === item.step ? 1.1 : 1,
                    }}
                    className="w-12 h-12 rounded-full flex items-center justify-center mb-2 shadow-lg"
                  >
                    {item.step === 1 && <ImageIcon className="w-5 h-5 text-white" />}
                    {item.step === 2 && <Cpu className="w-5 h-5 text-white" />}
                    {item.step === 3 && <ArrowRight className="w-5 h-5 text-white" />}
                  </motion.div>
                  <div className={pipelineStep >= item.step ? "text-green-400" : "text-slate-500"}>
                    {item.label}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">{item.desc}</div>
                </div>
              ))}
            </div>
            
            {/* Thanh kết nối */}
            <div className="absolute top-12 left-12 right-12 h-1 bg-slate-800 -z-0">
              <motion.div 
                className="h-full bg-green-500"
                initial={{ width: "0%" }}
                animate={{ width: pipelineStep === 0 ? "0%" : pipelineStep === 1 ? "25%" : pipelineStep === 2 ? "75%" : "100%" }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>

          {/* Results Card */}
          <AnimatePresence>
            {results && (
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
                  {results.map((res, idx) => (
                    <div key={idx}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-slate-700 truncate max-w-[70%]">
                          {res.className.replace(/___/g, ' - ').replace(/_/g, ' ')}
                        </span>
                        <span className="font-bold text-green-600">
                          {(res.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${res.confidence * 100}%` }}
                          transition={{ duration: 1, delay: idx * 0.1 }}
                          className={`h-full rounded-full ${idx === 0 ? 'bg-green-500' : 'bg-green-300'}`}
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
    </div>
  );
}
