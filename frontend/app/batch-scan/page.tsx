"use client";

import React, { useState, useRef } from "react";
import { Upload, Activity, ShieldAlert, CheckCircle2, AlertTriangle, Layers, X, Pill, Microscope } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runEdgeInference, getInferenceSession } from "../../lib/onnx-inference";
import { runCloudInference, CloudPredictionResult } from "../../lib/api";

type ProcessStatus = 'idle' | 'processing' | 'done' | 'error';
type HealthStatus = 'healthy' | 'mild' | 'severe';

interface ScannedImage {
  id: string;
  file: File;
  previewUrl: string;
  status: ProcessStatus;
  result?: {
    className: string;
    confidence: number;
    health: HealthStatus;
  };
}

export default function BatchScanPage() {
  const [images, setImages] = useState<ScannedImage[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Drill-down State
  const [selectedImage, setSelectedImage] = useState<ScannedImage | null>(null);
  const [cloudResult, setCloudResult] = useState<CloudPredictionResult | null>(null);
  const [isCloudProcessing, setIsCloudProcessing] = useState(false);

  // Epidemiological Stats
  const [stats, setStats] = useState({
    total: 0,
    diseased: 0,
    incidenceRate: 0,
    outbreakRisk: 'BÌNH THƯỜNG',
    clusterSummary: ''
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files).slice(0, 20); // Max 20 files
      
      const newImages = newFiles.map(file => ({
        id: Math.random().toString(36).substring(7),
        file,
        previewUrl: URL.createObjectURL(file), // Will revoke after inference
        status: 'idle' as ProcessStatus
      }));

      setImages(prev => [...prev, ...newImages]);
      setStats({ total: 0, diseased: 0, incidenceRate: 0, outbreakRisk: 'BÌNH THƯỜNG', clusterSummary: '' });
    }
  };

  const calculateHealthStatus = (className: string): HealthStatus => {
    if (className.toLowerCase().includes('healthy')) return 'healthy';
    // Đơn giản hóa: coi các bệnh là severe, có thể tinh chỉnh theo danh sách bệnh cụ thể
    return 'severe';
  };

  const runSequentialScan = async () => {
    if (images.length === 0) return;
    
    setIsScanning(true);
    setProgress({ current: 0, total: images.length });
    
    try {
      // 1. Tải trước Session để không bị delay giữa chừng
      await getInferenceSession();
      
      const results: Record<string, number> = {};
      let diseasedCount = 0;

      const updatedImages = [...images];

      // 2. Vòng lặp xử lý tuần tự (Sequential) để bảo vệ RAM
      for (let i = 0; i < updatedImages.length; i++) {
        const item = updatedImages[i];
        
        // Cập nhật trạng thái đang xử lý
        item.status = 'processing';
        setImages([...updatedImages]);
        
        try {
          // Gọi hàm Edge Inference (Tái sử dụng session bên dưới)
          const top5 = await runEdgeInference(item.file);
          const topClass = top5[0].className;
          const health = calculateHealthStatus(topClass);
          
          item.status = 'done';
          item.result = {
            className: topClass,
            confidence: top5[0].confidence,
            health
          };

          // Thống kê bệnh
          if (health !== 'healthy') {
            diseasedCount++;
            results[topClass] = (results[topClass] || 0) + 1;
          }
          
        } catch (err) {
          console.error("Lỗi khi xử lý ảnh", item.file.name, err);
          item.status = 'error';
        }
        
        // Ghi chú: RAM của ảnh gốc được giải phóng an toàn bên trong hàm `preprocessImage` 
        // của onnx-inference.ts (nơi chứa URL.revokeObjectURL() ngay sau khi vẽ Canvas).
        // previewUrl của UI sẽ được giữ lại để hiển thị Grid.

        setProgress({ current: i + 1, total: updatedImages.length });
        setImages([...updatedImages]);
        
        // Đợi 1 chút để nhường luồng UI render
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      // 3. Tính toán Báo cáo Dịch tễ học (Epidemiological Summary)
      const total = updatedImages.length;
      const rate = total > 0 ? (diseasedCount / total) * 100 : 0;
      
      let risk = 'BÌNH THƯỜNG';
      if (rate > 30) risk = 'CAO';
      else if (rate >= 10) risk = 'CẢNH BÁO';

      // Tìm cụm bệnh phổ biến nhất
      let maxDisease = '';
      let maxCount = 0;
      for (const [disease, count] of Object.entries(results)) {
        if (count > maxCount) {
          maxCount = count;
          maxDisease = disease;
        }
      }
      
      const clusterText = maxDisease ? `${((maxCount / diseasedCount) * 100).toFixed(0)}% số lá mắc bệnh là ${maxDisease.split('___').pop()?.replace(/_/g, ' ')}` : 'Không phát hiện cụm bệnh.';

      setStats({
        total,
        diseased: diseasedCount,
        incidenceRate: rate,
        outbreakRisk: risk,
        clusterSummary: clusterText
      });

    } catch (err) {
      console.error("Lỗi Batch Scan:", err);
      alert("Đã xảy ra lỗi trong quá trình quét. Vui lòng kiểm tra lại log.");
    } finally {
      setIsScanning(false);
    }
  };

  const handleCardClick = async (item: ScannedImage) => {
    if (item.status !== 'done') return;
    
    setSelectedImage(item);
    setCloudResult(null);
    setIsCloudProcessing(true);

    try {
      const data = await runCloudInference(item.file, 'efficientnetb0');
      setCloudResult(data);
    } catch (err) {
      console.error("Lỗi Cloud API:", err);
      alert("Không thể kết nối đến Cloud AI chuyên gia.");
    } finally {
      setIsCloudProcessing(false);
    }
  };

  const formatDiseaseName = (rawClass: string) => {
    const parts = rawClass.split('___');
    if (parts.length === 2) {
      return `${parts[1].replace(/_/g, ' ')} (${parts[0].replace(/_/g, ' ')})`;
    }
    return rawClass.replace(/_/g, ' ');
  };

  const getBorderColor = (status: ProcessStatus, health?: HealthStatus) => {
    if (status === 'processing') return 'border-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.5)]';
    if (status === 'error') return 'border-red-600 opacity-50';
    if (status === 'done' && health) {
      if (health === 'healthy') return 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]';
      if (health === 'mild') return 'border-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.3)]';
      return 'border-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.3)]';
    }
    return 'border-slate-200';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-6 pb-24">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-4 mb-10 pt-8 relative">
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 flex items-center justify-center gap-4">
            <Layers className="w-10 h-10 text-indigo-600" />
            Field-Level Batch Scanner
          </h1>
          <p className="text-slate-500 text-lg">
            Xử lý hàng loạt ảnh ngoại tuyến bằng Edge Computing. Không gửi dữ liệu lên máy chủ.
          </p>
        </div>

        {/* Dashboard Thống kê (Chỉ hiện khi quét xong ít nhất 1 lần) */}
        <AnimatePresence>
          {stats.total > 0 && !isScanning && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="bg-white border border-slate-200 p-6 rounded-3xl shadow-sm mb-8 grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="space-y-1 border-r border-slate-100 pr-4">
                <p className="text-sm font-semibold text-slate-500">Mức độ lây lan (Incidence Rate)</p>
                <h3 className="text-3xl font-black text-slate-800">{stats.incidenceRate.toFixed(1)}%</h3>
                <p className="text-xs text-slate-400">{stats.diseased} / {stats.total} mẫu nhiễm bệnh</p>
              </div>
              <div className="space-y-1 border-r border-slate-100 pr-4">
                <p className="text-sm font-semibold text-slate-500">Nguy cơ bùng phát (Risk)</p>
                <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-bold ${
                  stats.outbreakRisk === 'CAO' ? 'bg-red-100 text-red-700' : 
                  stats.outbreakRisk === 'CẢNH BÁO' ? 'bg-amber-100 text-amber-700' : 
                  'bg-emerald-100 text-emerald-700'
                }`}>
                  {stats.outbreakRisk === 'CAO' && <AlertTriangle className="w-4 h-4" />}
                  {stats.outbreakRisk === 'CẢNH BÁO' && <AlertTriangle className="w-4 h-4" />}
                  {stats.outbreakRisk === 'BÌNH THƯỜNG' && <CheckCircle2 className="w-4 h-4" />}
                  {stats.outbreakRisk}
                </div>
              </div>
              <div className="space-y-1 md:col-span-2">
                <p className="text-sm font-semibold text-slate-500">Phân vùng Cụm (Cluster Analysis)</p>
                <h4 className="text-lg font-bold text-slate-800 leading-tight">
                  {stats.clusterSummary}
                </h4>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Dropzone & Control */}
        <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-center gap-6">
          <div 
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 border-2 border-dashed border-indigo-200 hover:border-indigo-400 bg-indigo-50/50 hover:bg-indigo-50 transition-colors rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer text-center group"
          >
            <input 
              type="file" 
              multiple 
              accept="image/*" 
              className="hidden" 
              ref={fileInputRef} 
              onChange={handleFileChange}
            />
            <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center shadow-sm text-indigo-500 group-hover:scale-110 transition-transform mb-4">
              <Upload className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">Kéo thả hoặc chọn nhiều ảnh (Tối đa 20)</h3>
            <p className="text-slate-500 text-sm mt-1">Hỗ trợ JPG, PNG (Edge Computing trực tiếp trên trình duyệt)</p>
          </div>

          <div className="w-full md:w-64 space-y-4">
            <button 
              onClick={runSequentialScan} 
              disabled={isScanning || images.length === 0}
              className="w-full flex items-center justify-center p-4 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 transition-all font-semibold shadow-lg disabled:opacity-50 gap-3"
            >
              {isScanning ? <Activity className="w-5 h-5 animate-spin" /> : <ShieldAlert className="w-5 h-5"/>}
              {isScanning ? 'Đang phân tích...' : 'Bắt đầu Quét Đám'}
            </button>
            
            {/* Progress */}
            {isScanning && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm font-bold text-slate-700">
                  <span>Tiến độ quét:</span>
                  <span>{progress.current} / {progress.total}</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-indigo-500 transition-all duration-300"
                    style={{ width: `${(progress.current / progress.total) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Grid Hình ảnh */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {images.map((item, idx) => (
            <div 
              key={item.id} 
              onClick={() => handleCardClick(item)}
              className={`relative rounded-2xl overflow-hidden border-4 transition-all duration-300 bg-white ${getBorderColor(item.status, item.result?.health)} ${item.status === 'done' ? 'cursor-pointer hover:scale-[1.03] hover:shadow-lg hover:z-10' : ''}`}
            >
              <div className="aspect-square relative group">
                <img 
                  src={item.status === 'done' || item.status === 'error' ? (item.previewUrl || '/placeholder.png') : item.previewUrl} 
                  className={`w-full h-full object-cover transition-opacity ${item.status === 'processing' ? 'opacity-50 blur-sm' : 'opacity-100'}`} 
                  alt="Scan item" 
                />
                
                {item.status === 'processing' && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Activity className="w-8 h-8 text-blue-500 animate-spin" />
                  </div>
                )}
                
                {/* Result Overlay */}
                {item.status === 'done' && item.result && (
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-900/90 to-transparent p-3 pt-8 group-hover:from-slate-900 transition-colors">
                    <p className="text-white text-xs font-bold truncate">
                      {item.result.className.split('___').pop()?.replace(/_/g, ' ')}
                    </p>
                    <p className="text-slate-300 text-[10px]">
                      {(item.result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

      </div>

      {/* Drill-Down Clinical Inspection Dialog (Overlay) */}
      <AnimatePresence>
        {selectedImage && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }} 
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
              onClick={() => setSelectedImage(null)}
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }} 
              animate={{ opacity: 1, scale: 1, y: 0 }} 
              exit={{ opacity: 0, scale: 0.95, y: 20 }} 
              className="relative w-full max-w-3xl bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
            >
              {/* Dialog Header */}
              <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
                <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <Microscope className="w-5 h-5 text-indigo-600" />
                  Clinical Inspection Report
                </h3>
                <button 
                  onClick={() => setSelectedImage(null)}
                  className="p-2 rounded-full hover:bg-slate-200 text-slate-500 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Dialog Body */}
              <div className="p-6 overflow-y-auto flex-1">
                {isCloudProcessing ? (
                  <div className="flex flex-col items-center justify-center py-12 space-y-4">
                    <div className="w-16 h-16 rounded-full border-4 border-indigo-100 border-t-indigo-600 animate-spin"></div>
                    <p className="text-slate-600 font-medium">Đang kết nối chuyên gia AI (Cloud)...</p>
                    <p className="text-sm text-slate-400">Trích xuất Grad-CAM và Phác đồ điều trị</p>
                  </div>
                ) : cloudResult ? (
                  <div className="space-y-8">
                    {/* Header Info */}
                    <div className="flex flex-col md:flex-row gap-6 items-start">
                      <div className="w-32 h-32 shrink-0 rounded-2xl overflow-hidden border-4 border-slate-100 shadow-sm relative">
                         <img src={`data:image/jpeg;base64,${cloudResult.xai.overlay_b64}`} className="w-full h-full object-cover" alt="XAI Overlay" />
                         <div className="absolute top-1 left-1 bg-slate-900/60 text-white text-[10px] px-1.5 py-0.5 rounded backdrop-blur-sm">Grad-CAM</div>
                      </div>
                      <div className="space-y-2 flex-1">
                        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-medium text-xs mb-1">
                          <CheckCircle2 className="w-3 h-3" />
                          Độ tự tin: {(cloudResult.confidence * 100).toFixed(1)}%
                        </div>
                        <h2 className="text-2xl font-extrabold text-slate-900">
                          {formatDiseaseName(cloudResult.predicted_class)}
                        </h2>
                        
                        {/* Severity Bar */}
                        <div className="mt-4 pt-4 border-t border-slate-100">
                          <div className="flex justify-between text-sm font-bold mb-1">
                            <span className="text-slate-600">Mức độ nghiêm trọng (Severity)</span>
                            <span className={cloudResult.xai.severity.level === 'Severe' ? 'text-red-600' : 'text-amber-600'}>
                              {cloudResult.xai.severity.level} ({cloudResult.xai.severity.percentage}%)
                            </span>
                          </div>
                          <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-500 ${cloudResult.xai.severity.level === 'Severe' ? 'bg-red-500' : cloudResult.xai.severity.level === 'Moderate' ? 'bg-amber-500' : 'bg-emerald-500'}`}
                              style={{ width: `${cloudResult.xai.severity.percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Treatment Protocol */}
                    <div className="bg-blue-50 border border-blue-100 p-5 rounded-2xl">
                      <h4 className="text-blue-900 font-bold flex items-center gap-2 mb-3">
                        <Pill className="w-5 h-5" /> Phác đồ Điều trị
                      </h4>
                      <p className="text-blue-800 font-medium mb-2">{cloudResult.recommendation.action}</p>
                      <p className="text-blue-700/80 text-sm leading-relaxed">{cloudResult.recommendation.description}</p>
                    </div>

                  </div>
                ) : null}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
