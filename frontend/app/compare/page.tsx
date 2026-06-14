"use client";

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Zap, Cloud, Activity, FileText, Database, Shield, Image as ImageIcon } from "lucide-react";
import { runEdgeInference } from "@/lib/onnx-inference";
import { runCloudInference, CloudPredictionResult } from "@/lib/api";
import dynamic from "next/dynamic";

// Dynamically import Recharts to prevent SSR Hydration Mismatches
const RadarChart = dynamic(() => import("recharts").then(mod => mod.RadarChart), { ssr: false });
const Radar = dynamic(() => import("recharts").then(mod => mod.Radar), { ssr: false });
const PolarGrid = dynamic(() => import("recharts").then(mod => mod.PolarGrid), { ssr: false });
const PolarAngleAxis = dynamic(() => import("recharts").then(mod => mod.PolarAngleAxis), { ssr: false });
const PolarRadiusAxis = dynamic(() => import("recharts").then(mod => mod.PolarRadiusAxis), { ssr: false });
const ResponsiveContainer = dynamic(() => import("recharts").then(mod => mod.ResponsiveContainer), { ssr: false });

// Static metrics for comparison
const staticMetrics = [
  { metric: "Accuracy (%)", Edge: 94.2, Cloud: 98.7, fullMark: 100 },
  { metric: "Macro F1 (%)", Edge: 92.5, Cloud: 98.2, fullMark: 100 },
  { metric: "Params (M)", Edge: 10, Cloud: 25, fullMark: 30 }, // Inverted logic for radar visually if needed, but we keep raw here
  { metric: "Size (MB)", Edge: 16, Cloud: 65, fullMark: 100 },
];

export default function ComparePage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  
  // Results state
  const [edgeResult, setEdgeResult] = useState<{class: string, confidence: number, time: number} | null>(null);
  const [cloudResult, setCloudResult] = useState<{class: string, confidence: number, time: number} | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      
      // Reset results when new image is uploaded
      setEdgeResult(null);
      setCloudResult(null);
    }
  };

  const handleCompare = async () => {
    if (!image) return;
    setIsComparing(true);
    setEdgeResult(null);
    setCloudResult(null);

    try {
      // Create promises for concurrent execution
      const edgeStartTime = performance.now();
      const edgePromise = runEdgeInference(image).then((results) => {
        const edgeEndTime = performance.now();
        return {
          class: results[0].className,
          confidence: results[0].probability * 100,
          time: edgeEndTime - edgeStartTime
        };
      });

      const cloudStartTime = performance.now();
      const cloudPromise = runCloudInference(image, "efficientnetb0").then((result) => {
        const cloudEndTime = performance.now();
        return {
          class: result.predicted_class,
          confidence: result.confidence * 100,
          time: cloudEndTime - cloudStartTime
        };
      });

      // Run concurrently
      const [edgeRes, cloudRes] = await Promise.all([edgePromise, cloudPromise]);
      
      setEdgeResult(edgeRes);
      setCloudResult(cloudRes);
    } catch (error) {
      console.error("Error during comparison:", error);
      alert("Đã xảy ra lỗi trong quá trình so sánh. Vui lòng kiểm tra console.");
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-[#0a192f] to-slate-900 text-white p-6 pb-24">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-blue-500">
            Deep Learning Battle Center
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Khám phá sự đánh đổi giữa Tốc độ siêu việt (Edge) và Độ chính xác tuyệt đối (Cloud) thông qua bài kiểm tra thời gian thực.
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-slate-800/50 backdrop-blur-md rounded-3xl p-8 border border-slate-700/50 shadow-2xl flex flex-col items-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
            id="compare-upload"
          />
          <label
            htmlFor="compare-upload"
            className="cursor-pointer group flex flex-col items-center gap-4"
          >
            <div className="w-20 h-20 rounded-full bg-slate-700/50 flex items-center justify-center group-hover:bg-teal-500/20 transition-colors border-2 border-dashed border-slate-500 group-hover:border-teal-400">
              <Upload className="w-10 h-10 text-slate-400 group-hover:text-teal-400 transition-colors" />
            </div>
            <span className="text-lg font-medium text-slate-300 group-hover:text-teal-400 transition-colors">
              {preview ? "Thay đổi ảnh kiểm thử" : "Tải ảnh lên để bắt đầu trận chiến"}
            </span>
          </label>

          {preview && (
            <div className="mt-8 flex flex-col items-center gap-6">
              <div className="relative w-64 h-64 rounded-2xl overflow-hidden shadow-2xl border-2 border-slate-600">
                <img src={preview} alt="Test" className="w-full h-full object-cover" />
              </div>
              <button
                onClick={handleCompare}
                disabled={isComparing}
                className="px-8 py-3 bg-gradient-to-r from-teal-500 to-blue-600 rounded-xl font-bold text-lg shadow-lg hover:shadow-teal-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isComparing ? (
                  <>
                    <Activity className="w-5 h-5 animate-spin" />
                    Đang giao tranh...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Bắt đầu So sánh
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Battle Arena */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* MobileNetV3 (Edge) */}
          <div className="bg-slate-800/50 backdrop-blur-md rounded-3xl p-6 border border-slate-700/50 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Zap className="w-32 h-32 text-amber-500" />
            </div>
            <div className="relative z-10 space-y-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-amber-500/20 rounded-xl">
                  <Zap className="w-8 h-8 text-amber-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-100">MobileNetV3</h2>
                  <p className="text-amber-400/80 font-medium">Edge Computing (Trình duyệt)</p>
                </div>
              </div>

              <div className="min-h-[120px] bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
                {edgeResult ? (
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-slate-400 mb-1">Dự đoán</p>
                      <p className="text-xl font-bold text-emerald-400">{edgeResult.class}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-400 mb-1">Độ tự tin</p>
                        <p className="text-lg font-semibold text-slate-200">{edgeResult.confidence.toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-sm text-slate-400 mb-1">Thời gian (Thực tế)</p>
                        <p className="text-lg font-semibold text-amber-400">{edgeResult.time.toFixed(0)} ms</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500">
                    {isComparing ? "Đang xử lý cục bộ..." : "Đang chờ dữ liệu..."}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* EfficientNetB0 (Cloud) */}
          <div className="bg-slate-800/50 backdrop-blur-md rounded-3xl p-6 border border-slate-700/50 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Cloud className="w-32 h-32 text-blue-500" />
            </div>
            <div className="relative z-10 space-y-6">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-blue-500/20 rounded-xl">
                  <Cloud className="w-8 h-8 text-blue-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-100">EfficientNetB0</h2>
                  <p className="text-blue-400/80 font-medium">Cloud API (GPU Server)</p>
                </div>
              </div>

              <div className="min-h-[120px] bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50">
                {cloudResult ? (
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-slate-400 mb-1">Dự đoán</p>
                      <p className="text-xl font-bold text-emerald-400">{cloudResult.class}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-400 mb-1">Độ tự tin</p>
                        <p className="text-lg font-semibold text-slate-200">{cloudResult.confidence.toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-sm text-slate-400 mb-1">Thời gian (Thực tế + Mạng)</p>
                        <p className="text-lg font-semibold text-blue-400">{cloudResult.time.toFixed(0)} ms</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500">
                    {isComparing ? "Đang gọi API..." : "Đang chờ dữ liệu..."}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Static Metrics Comparison */}
        <div className="bg-slate-800/50 backdrop-blur-md rounded-3xl p-8 border border-slate-700/50 shadow-2xl">
          <h3 className="text-2xl font-bold mb-6 text-slate-100 flex items-center gap-3">
            <Shield className="w-6 h-6 text-teal-400" />
            Đánh giá Kiến trúc (Static Metrics)
          </h3>
          
          <div className="grid md:grid-cols-2 gap-8 items-center">
            {/* Radar Chart */}
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={staticMetrics}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 14 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="MobileNetV3 (Edge)" dataKey="Edge" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.5} />
                  <Radar name="EfficientNetB0 (Cloud)" dataKey="Cloud" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                </RadarChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-6 mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                  <span className="text-sm text-slate-300">MobileNetV3</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span className="text-sm text-slate-300">EfficientNetB0</span>
                </div>
              </div>
            </div>

            {/* Confusion Matrix Placeholders */}
            <div className="space-y-6">
              <div className="space-y-2">
                <p className="text-sm font-medium text-slate-400 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4" /> Ma trận nhầm lẫn (MobileNetV3)
                </p>
                <div className="h-32 bg-slate-900/80 rounded-xl border border-slate-700 flex items-center justify-center overflow-hidden">
                  <span className="text-slate-600 text-sm">Chưa có ảnh (Placeholder)</span>
                </div>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-slate-400 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4" /> Ma trận nhầm lẫn (EfficientNetB0)
                </p>
                <div className="h-32 bg-slate-900/80 rounded-xl border border-slate-700 flex items-center justify-center overflow-hidden">
                  <span className="text-slate-600 text-sm">Chưa có ảnh (Placeholder)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
