"use client";

import React, { useState } from "react";
import {
  Upload, Cloud, Activity, AlertTriangle,
  ShieldAlert, Beaker, Microscope, Pill, Stethoscope, Image as ImageIcon
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runCloudInference, CloudPredictionResult } from "@/lib/api";
import dynamic from "next/dynamic";

// Dynamic Imports to prevent Hydration Mismatch
const PieChart = dynamic(() => import("recharts").then(mod => mod.PieChart), { ssr: false });
const Pie = dynamic(() => import("recharts").then(mod => mod.Pie), { ssr: false });
const Cell = dynamic(() => import("recharts").then(mod => mod.Cell), { ssr: false });
const ResponsiveContainer = dynamic(() => import("recharts").then(mod => mod.ResponsiveContainer), { ssr: false });
const BarChart = dynamic(() => import("recharts").then(mod => mod.BarChart), { ssr: false });
const Bar = dynamic(() => import("recharts").then(mod => mod.Bar), { ssr: false });
const XAxis = dynamic(() => import("recharts").then(mod => mod.XAxis), { ssr: false });
const YAxis = dynamic(() => import("recharts").then(mod => mod.YAxis), { ssr: false });
const Tooltip = dynamic(() => import("recharts").then(mod => mod.Tooltip), { ssr: false });

export default function DiagnosticReportPage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  // States
  const [model, setModel] = useState<'mobilenetv3' | 'efficientnetb0'>('mobilenetv3');
  const [isInferring, setIsInferring] = useState(false);
  const [cloudResult, setCloudResult] = useState<CloudPredictionResult | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (preview) URL.revokeObjectURL(preview); // memory leak fix
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setCloudResult(null);
    }
  };

  const runCloud = async () => {
    if (!image) return;
    setIsInferring(true);
    setCloudResult(null);

    try {
      const start = performance.now();
      const res = await runCloudInference(image, model);
      const end = performance.now();
      setInferenceTime(end - start);
      setCloudResult(res);
    } catch (e) {
      alert("Lỗi khi gọi API Cloud!");
    } finally {
      setIsInferring(false);
    }
  };

  // Severity Component UI
  const renderSeverityGauge = (severityPercentage: number, severityLevel: string) => {
    let color = "#10b981"; // Mild / Green
    if (severityLevel === "Moderate") color = "#f59e0b"; // Orange
    if (severityLevel === "Severe") color = "#ef4444"; // Red

    const data = [
      { name: "Affected", value: severityPercentage, fill: color },
      { name: "Healthy", value: 100 - severityPercentage, fill: "#e2e8f0" }
    ];

    return (
      <div className="relative w-40 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              innerRadius={50}
              outerRadius={70}
              startAngle={180}
              endAngle={0}
              dataKey="value"
              stroke="none"
              cornerRadius={5}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-8">
          <span className="text-2xl font-bold" style={{ color }}>
            {severityPercentage}%
          </span>
          <span className="text-xs text-slate-500 uppercase tracking-widest mt-1">
            {severityLevel}
          </span>
        </div>
      </div>
    );
  };

  const [viewMode, setViewMode] = useState<'standard' | 'expert'>('expert');

  const formatDiseaseName = (rawClass: string) => {
    const parts = rawClass.split('___');
    if (parts.length === 2) {
      const plant = parts[0].replace(/_/g, ' ');
      const disease = parts[1].replace(/_/g, ' ');
      return `${disease} (${plant})`;
    }
    return rawClass.replace(/_/g, ' ');
  };

  const renderSkeleton = () => (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full animate-pulse mt-8">
      {viewMode === 'expert' && (
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white rounded-3xl h-[600px] border border-slate-200 shadow-sm"></div>
        </div>
      )}
      <div className={`lg:col-span-${viewMode === 'expert' ? '8' : '12'} space-y-6`}>
        <div className="bg-white rounded-3xl h-64 border border-slate-200 shadow-sm"></div>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white rounded-3xl h-64 border border-slate-200 shadow-sm"></div>
          <div className="bg-white rounded-3xl h-64 border border-slate-200 shadow-sm"></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-6 pb-24">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="text-center space-y-4 mb-10 pt-8 relative">
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 flex items-center justify-center gap-4">
            <Stethoscope className="w-10 h-10 text-teal-600" />
            Phòng Chẩn đoán (Clinical Lab)
          </h1>
          <p className="text-slate-500 text-lg">
            Hệ thống phân tích bệnh học độ phân giải cao dành cho kỹ sư nông nghiệp.
          </p>

          {/* Mode Switcher */}
          <div className="flex justify-center mt-6">
            <div className="bg-white p-1.5 rounded-full border border-slate-200 inline-flex shadow-sm">
              <button
                onClick={() => setViewMode('standard')}
                className={`px-6 py-2 rounded-full font-medium text-sm transition-all ${viewMode === 'standard'
                    ? 'bg-slate-900 text-white shadow'
                    : 'text-slate-500 hover:text-slate-800'
                  }`}
              >
                Chế độ Nông dân (Standard)
              </button>
              <button
                onClick={() => setViewMode('expert')}
                className={`px-6 py-2 rounded-full font-medium text-sm transition-all ${viewMode === 'expert'
                    ? 'bg-slate-900 text-white shadow'
                    : 'text-slate-500 hover:text-slate-800'
                  }`}
              >
                Chế độ Kỹ sư (Expert)
              </button>
            </div>
          </div>
        </div>

        {/* Action Panel */}
        <div className="bg-white rounded-3xl p-8 border border-slate-200 shadow-sm max-w-3xl mx-auto">
          {!preview ? (
            <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-slate-300 rounded-2xl hover:bg-slate-50 transition-colors cursor-pointer relative">
              <input type="file" accept="image/*" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer" />
              <Upload className="w-12 h-12 text-slate-400 mb-4" />
              <p className="text-lg font-medium text-slate-600">Tải ảnh mẫu vật (Lá cây) lên để phân tích</p>
            </div>
          ) : (
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="relative w-32 h-32 md:w-48 md:h-48 rounded-2xl overflow-hidden border-2 border-slate-200 shrink-0 shadow-sm">
                <img src={preview} alt="Sample" className="w-full h-full object-cover" />
                <button onClick={() => { setPreview(null); setImage(null); }} className="absolute top-2 right-2 bg-slate-900/60 text-white p-1.5 rounded-lg hover:bg-red-500 transition-colors">
                  ✕
                </button>
              </div>

              <div className="flex-1 flex flex-col gap-6 w-full">
                {/* Model Selector */}
                <div className="bg-slate-100 p-2 rounded-2xl flex gap-2">
                  <button
                    onClick={() => setModel('mobilenetv3')}
                    className={`flex-1 py-3 rounded-xl font-semibold transition-all ${model === 'mobilenetv3'
                        ? 'bg-white text-teal-600 shadow-sm border border-slate-200/50'
                        : 'text-slate-500 hover:text-slate-700'
                      }`}
                  >
                    MobileNetV3
                  </button>
                  <button
                    onClick={() => setModel('efficientnetb0')}
                    className={`flex-1 py-3 rounded-xl font-semibold transition-all ${model === 'efficientnetb0'
                        ? 'bg-white text-indigo-600 shadow-sm border border-slate-200/50'
                        : 'text-slate-500 hover:text-slate-700'
                      }`}
                  >
                    EfficientNetB0
                  </button>
                </div>

                <button
                  onClick={runCloud}
                  disabled={isInferring}
                  className="w-full flex items-center justify-center p-4 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-all font-semibold shadow-lg disabled:opacity-50 gap-3"
                >
                  {isInferring ? <Activity className="w-5 h-5 animate-spin" /> : <Microscope className="w-5 h-5" />}
                  {isInferring ? 'Đang phân tích chuyên sâu...' : 'Thực hiện Chẩn đoán (Deep Clinical)'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Loading State */}
        {isInferring && renderSkeleton()}

        {/* ----------------- CLOUD RESULTS (MEDICAL REPORT) ----------------- */}
        {!isInferring && cloudResult && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6 mt-8">

            {/* Uncertainty Alert */}
            {cloudResult.uncertainty_warning && (
              <div className="bg-red-50 border border-red-200 p-5 rounded-2xl flex items-start gap-4 shadow-sm">
                <AlertTriangle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-red-700 font-bold text-lg">Cảnh báo: Độ chắc chắn thấp (High Entropy)</h3>
                  <p className="text-red-600/80 mt-1">Khoảng cách giữa bệnh khả năng cao nhất và thứ hai rất thấp. Khuyến cáo chụp thêm ảnh ở góc độ khác hoặc tham vấn trực tiếp chuyên gia nông nghiệp để tránh chẩn đoán sai.</p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

              {/* CỘT TRÁI: AI EXECUTION TRACE (Chỉ hiện trong Expert Mode) */}
              {viewMode === 'expert' && (
                <div className="lg:col-span-4 space-y-6">
                  <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm h-full">
                    <h3 className="text-xl font-bold text-slate-800 mb-8 flex items-center gap-2">
                      <Activity className="text-teal-500" /> Dấu vết AI (Pipeline Trace)
                    </h3>

                    <div className="space-y-10 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-slate-200">
                      {/* Step 1: TTA Batching */}
                      <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                        <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-slate-100 text-slate-500 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10 shadow-sm">
                          <ImageIcon className="w-4 h-4" />
                        </div>
                        <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
                          <p className="text-sm font-bold text-slate-600">1. Multi-View TTA</p>
                          <p className="text-xs text-slate-500 mb-3">(Ensemble Input 4 Góc độ)</p>
                          <div className="flex flex-col gap-3">
                            {cloudResult.xai.tta_b64_list.map((b64, idx) => (
                              <div key={idx} className="relative">
                                <span className="absolute top-1 left-1 bg-slate-900/60 text-white text-[10px] px-1.5 py-0.5 rounded backdrop-blur-sm">View {idx + 1}</span>
                                <img src={`data:image/jpeg;base64,${b64}`} alt={`TTA View ${idx + 1}`} className="w-full rounded-lg object-cover border border-slate-200 shadow-sm" />
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Step 2: Conv1 Feature Map */}
                      <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                        <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-teal-50 text-teal-600 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10 shadow-sm">
                          <Cloud className="w-4 h-4" />
                        </div>
                        <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
                          <p className="text-sm font-bold text-teal-600">2. Low-level Feature (Conv1)</p>
                          <p className="text-xs text-slate-500 mb-3">(Phát hiện viền/mảng màu - Channel 0)</p>
                          <img src={`data:image/jpeg;base64,${cloudResult.xai.feature_map_b64}`} className="w-full rounded-xl h-28 object-cover border border-slate-100" />
                        </div>
                      </div>

                      {/* Step 3: Leaf Mask */}
                      <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                        <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-indigo-50 text-indigo-600 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10 shadow-sm">
                          <Beaker className="w-4 h-4" />
                        </div>
                        <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
                          <p className="text-sm font-bold text-indigo-600">3. Background Suppression</p>
                          <p className="text-xs text-slate-500 mb-3">(Phân tách nền bằng Mask Trắng Đen)</p>
                          <img src={`data:image/jpeg;base64,${cloudResult.xai.leaf_mask_b64}`} className="w-full rounded-xl h-28 object-cover border border-slate-100 bg-slate-900" />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}


              {/* CỘT PHẢI: LÂM SÀNG VÀ ĐIỀU TRỊ (Mở rộng toàn màn hình nếu Standard Mode) */}
              <div className={`${viewMode === 'expert' ? 'lg:col-span-8' : 'lg:col-span-12'} transition-all duration-500`}>

                {/* Clinical Findings */}
                <div className="bg-white border border-slate-200 rounded-3xl p-6 md:p-8 shadow-sm">
                  <h3 className="text-xl font-bold text-slate-800 mb-8 flex items-center gap-2">
                    <ShieldAlert className="text-blue-500" /> Phân tích Lâm sàng (Clinical Findings)
                  </h3>
                  <div className="flex flex-col md:flex-row gap-8 items-center">
                    <div className="w-48 h-48 md:w-64 md:h-64 rounded-3xl overflow-hidden shrink-0 border border-slate-200 relative group shadow-sm">
                      <img src={`data:image/jpeg;base64,${cloudResult.xai.overlay_b64}`} className="w-full h-full object-cover transition-opacity duration-300" />
                      <div className="absolute inset-0 bg-slate-900/80 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity p-4 text-center">
                        <p className="text-sm text-white font-medium mb-3">Heatmap thô (Raw Grad-CAM)</p>
                        <img src={`data:image/jpeg;base64,${cloudResult.xai.heatmap_b64}`} className="w-28 h-28 rounded-xl object-cover shadow-lg border border-slate-700" />
                      </div>
                    </div>

                    <div className="flex-1 space-y-4 text-center md:text-left">
                      <h4 className="text-2xl md:text-3xl font-extrabold text-slate-900">
                        {formatDiseaseName(cloudResult.predicted_class)}
                      </h4>
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 font-medium text-sm">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        Độ tự tin: {(cloudResult.confidence * 100).toFixed(1)}%
                      </div>

                      <div className="pt-6 mt-6 border-t border-slate-100 flex flex-col sm:flex-row items-center gap-6">
                        {renderSeverityGauge(cloudResult.xai.severity.percentage, cloudResult.xai.severity.level)}
                        <div className="text-left">
                          <p className="text-slate-600 text-lg">Mức độ lây nhiễm: <strong className="text-slate-900">{cloudResult.xai.severity.level}</strong></p>
                          <p className="text-slate-500 text-sm mt-2 leading-relaxed">
                            Hệ thống đo lường có <span className="font-bold text-slate-700">{cloudResult.xai.severity.percentage}%</span> diện tích lá bị ảnh hưởng bởi vùng kích hoạt bệnh lý.
                          </p>
                          <p className="text-slate-400 text-xs mt-2 italic">
                            * Tính toán dựa trên giao tuyến không gian HSV và Grad-CAM mask.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Grid con cho Top-K và Protocol */}
                <div className="space-y-6">
                  {/* Differential Diagnosis */}
                  <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-800 mb-4">Chẩn đoán phân biệt (Top 5)</h3>
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={cloudResult.top5} layout="vertical" margin={{ top: 0, right: 20, left: -20, bottom: 0 }}>
                          <XAxis type="number" domain={[0, 1]} hide />
                          <YAxis dataKey="class" type="category" tick={{ fill: '#64748b', fontSize: 12, fontWeight: 500 }} width={120} tickFormatter={(val) => val.split('___').pop()} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '12px', color: '#0f172a', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} cursor={{ fill: '#f1f5f9' }} />
                          <Bar dataKey="confidence" fill="#3b82f6" radius={[0, 6, 6, 0]} barSize={24}>
                            {cloudResult.top5.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? '#10b981' : '#cbd5e1'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Treatment Protocol */}
                  <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm flex flex-col">
                    <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                      <Pill className="text-teal-500 w-5 h-5" /> Phác đồ Xử lý
                    </h3>
                    <div className="flex-1 bg-slate-50 rounded-2xl p-5 border border-slate-100 flex flex-col">
                      <p className="font-bold text-slate-800 mb-3 text-lg leading-tight">{cloudResult.recommendation.action}</p>
                      <p className="text-slate-600 text-sm leading-relaxed flex-1">
                        {cloudResult.recommendation.description}
                      </p>
                      <div className="mt-6 pt-4 border-t border-slate-200">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${cloudResult.recommendation.priority === 'high' ? 'bg-red-100 text-red-700' :
                            cloudResult.recommendation.priority === 'medium' ? 'bg-amber-100 text-amber-700' :
                              'bg-emerald-100 text-emerald-700'
                          }`}>
                          Ưu tiên: {cloudResult.recommendation.priority}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
