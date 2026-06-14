"use client";

import React, { useEffect, useState } from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from "recharts";
import { Activity, ShieldAlert, Thermometer, FlaskConical, Clock, Plus, Check, Sprout } from "lucide-react";
import { PredictionRecord, PlantRecord, getAllPredictions, addIntervention, getAllPlants } from "../lib/db";

export default function DashboardPage() {
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [plants, setPlants] = useState<PlantRecord[]>([]);
  const [selectedPlantId, setSelectedPlantId] = useState<string>("ALL");
  const [loading, setLoading] = useState(true);
  const [interventionInput, setInterventionInput] = useState<{ id: number, text: string } | null>(null);

  useEffect(() => {
    loadPlants();
  }, []);

  useEffect(() => {
    loadData();
  }, [selectedPlantId]);

  const loadPlants = async () => {
    try {
      const p = await getAllPlants();
      setPlants(p);
      if (p.length > 0 && selectedPlantId === "ALL") {
        setSelectedPlantId(p[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      // Nếu chọn "ALL" thì fetch tất cả, hoặc không filter. Ở đây ta ưu tiên xem từng cây.
      const data = await getAllPredictions(selectedPlantId === "ALL" ? undefined : selectedPlantId, 50);
      setHistory(data);
    } catch (err) {
      console.error("Failed to load history", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddIntervention = async (id: number) => {
    if (!interventionInput || !interventionInput.text.trim()) return;
    try {
      await addIntervention(id, interventionInput.text.trim());
      setInterventionInput(null);
      loadData(); // Refresh
    } catch (err) {
      console.error(err);
      alert("Failed to save intervention note");
    }
  };

  // Quick Stats Calculation
  const totalScans = history.length;
  const diseaseCounts: Record<string, number> = {};
  let maxDisease = "N/A";
  let maxCount = 0;
  
  history.forEach(item => {
    if (!item.predictedClass.toLowerCase().includes('healthy')) {
      diseaseCounts[item.predictedClass] = (diseaseCounts[item.predictedClass] || 0) + 1;
      if (diseaseCounts[item.predictedClass] > maxCount) {
        maxCount = diseaseCounts[item.predictedClass];
        maxDisease = item.predictedClass;
      }
    }
  });

  // Prepare Chart Data
  const chartData = history.map(item => ({
    ...item,
    formattedDate: new Date(item.createdAt).toLocaleDateString('vi-VN') + ' ' + new Date(item.createdAt).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }),
    severity: item.severityPercentage
  }));

  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (payload.intervention) {
      return (
        <g>
          <circle cx={cx} cy={cy} r={8} fill="#ef4444" stroke="#fee2e2" strokeWidth={3} />
          <text x={cx} y={cy - 15} textAnchor="middle" fill="#ef4444" fontSize={10} fontWeight="bold">Phun thuốc</text>
        </g>
      );
    }
    return <circle cx={cx} cy={cy} r={4} fill="#3b82f6" stroke="#ffffff" strokeWidth={1} />;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-slate-200 rounded-xl shadow-lg">
          <p className="text-slate-500 text-xs font-medium mb-1">{data.formattedDate}</p>
          <p className="text-slate-900 font-bold">{data.predictedClass.split('___').pop()?.replace(/_/g, ' ')}</p>
          <p className="text-amber-600 font-semibold mt-1">Mức độ bệnh: {data.severity}%</p>
          {data.intervention && (
            <div className="mt-2 pt-2 border-t border-slate-100 text-red-600 text-xs font-bold flex items-start gap-1">
              <FlaskConical className="w-3 h-3 mt-0.5" />
              <span>Can thiệp: {data.intervention}</span>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-6 lg:p-10 pb-24">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header & Plant Selector */}
        <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 flex items-center gap-3">
              <Activity className="w-8 h-8 text-indigo-600" />
              Longitudinal Treatment Tracker
            </h1>
            <p className="text-slate-500 mt-2 text-lg">
              Theo dõi diễn tiến bệnh học và đánh giá hiệu quả phác đồ điều trị theo thời gian thực.
            </p>
          </div>
          
          <div className="bg-white border border-slate-200 p-2 rounded-2xl shadow-sm flex items-center gap-3 min-w-[250px]">
            <div className="w-10 h-10 rounded-xl bg-teal-50 flex items-center justify-center shrink-0">
              <Sprout className="w-5 h-5 text-teal-600" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-0.5">Hồ sơ cây trồng</p>
              <select 
                value={selectedPlantId}
                onChange={(e) => setSelectedPlantId(e.target.value)}
                className="w-full text-slate-800 font-bold bg-transparent outline-none cursor-pointer"
              >
                {plants.length === 0 ? (
                  <option value="ALL">Chưa có hồ sơ nào</option>
                ) : (
                  <>
                    <option value="ALL">-- Tất cả hồ sơ --</option>
                    {plants.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </>
                )}
              </select>
            </div>
          </div>
        </div>

        {/* Khu vực 1: Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex items-center gap-5">
            <div className="w-14 h-14 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Clock className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-500">Tổng số lượt theo dõi</p>
              <h3 className="text-2xl font-black text-slate-900">{totalScans} <span className="text-sm font-normal text-slate-400">bản ghi</span></h3>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex items-center gap-5 md:col-span-2">
            <div className="w-14 h-14 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center">
              <ShieldAlert className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-500">Bệnh xuất hiện nhiều nhất</p>
              <h3 className="text-xl font-black text-slate-900">
                {maxDisease !== "N/A" ? maxDisease.split('___').pop()?.replace(/_/g, ' ') : "Chưa có dữ liệu"} 
                {maxCount > 0 && <span className="text-sm font-normal text-slate-400 ml-2">({maxCount} lần mắc)</span>}
              </h3>
            </div>
          </div>
        </div>

        {/* Khu vực 2: Treatment Efficacy Chart */}
        <div className="bg-white p-6 md:p-8 rounded-3xl border border-slate-200 shadow-sm">
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
            <div>
              <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                <Thermometer className="w-6 h-6 text-rose-500" /> Biểu đồ Diễn tiến Mức độ nghiêm trọng (Severity %)
              </h3>
              <p className="text-slate-500 text-sm mt-1">Điểm màu đỏ biểu thị các mốc thời gian có can thiệp/phun thuốc.</p>
            </div>
          </div>
          
          <div className="h-80 w-full">
            {history.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="formattedDate" 
                    tick={{ fill: '#64748b', fontSize: 12 }} 
                    axisLine={false} 
                    tickLine={false} 
                    tickFormatter={(val) => val.split(' ')[0]} // Only show date on axis
                  />
                  <YAxis 
                    domain={[0, 100]} 
                    tick={{ fill: '#64748b', fontSize: 12 }} 
                    axisLine={false} 
                    tickLine={false} 
                    tickFormatter={(val) => `${val}%`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="severity" 
                    stroke="#3b82f6" 
                    strokeWidth={3} 
                    dot={<CustomDot />} 
                    activeDot={{ r: 8, strokeWidth: 0, fill: '#1d4ed8' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="w-full h-full flex items-center justify-center border-2 border-dashed border-slate-200 rounded-2xl text-slate-400">
                Chưa có dữ liệu chẩn đoán để vẽ biểu đồ
              </div>
            )}
          </div>
        </div>

        {/* Khu vực 3: History Feed */}
        <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
           <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
            <Clock className="w-6 h-6 text-slate-400" /> Lịch sử Chẩn đoán & Can thiệp
          </h3>
          
          <div className="space-y-4">
            {history.length === 0 && (
              <p className="text-slate-500 text-center py-6">Không có dữ liệu lịch sử.</p>
            )}
            
            {/* Array reversed above, so newest is at the end? Oh wait, getAllPredictions reverses so oldest is first? 
                Actually, usually history feed wants newest first. Let's reverse a copy for the feed. */}
            {[...history].reverse().map(item => (
              <div key={item.id} className="p-4 border border-slate-100 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 hover:bg-slate-50 transition-colors">
                <div className="flex items-center gap-4">
                  {/* Thumbnail */}
                  <div className="w-16 h-16 shrink-0 rounded-xl overflow-hidden bg-slate-100 border border-slate-200">
                    {/* Convert Blob to URL (temporary for UI, memory leak is minor here but should ideally use createObjectURL cleanup) */}
                    {item.imageThumbnail && (
                      <img src={URL.createObjectURL(item.imageThumbnail)} alt="Thumbnail" className="w-full h-full object-cover" />
                    )}
                  </div>
                  
                  <div>
                    <h4 className="font-bold text-slate-800">{item.predictedClass.split('___').pop()?.replace(/_/g, ' ')}</h4>
                    <div className="flex items-center gap-3 mt-1 text-sm">
                      <span className="text-slate-500">{new Date(item.createdAt).toLocaleString('vi-VN')}</span>
                      <span className={`font-semibold ${item.severityPercentage > 50 ? 'text-red-600' : 'text-amber-600'}`}>
                        Mức độ: {item.severityPercentage}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Intervention Action */}
                <div className="w-full md:w-auto">
                  {item.intervention ? (
                    <div className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-red-50 text-red-700 border border-red-100 text-sm font-medium">
                      <FlaskConical className="w-4 h-4 shrink-0" />
                      <span>Ghi chú: {item.intervention}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      {interventionInput?.id === item.id ? (
                        <div className="flex items-center gap-2">
                          <input 
                            type="text" 
                            className="text-sm border border-slate-300 rounded-lg px-3 py-2 outline-none focus:border-indigo-500 w-full md:w-48"
                            placeholder="Nhập loại thuốc phun..."
                            autoFocus
                            value={interventionInput.text}
                            onChange={(e) => setInterventionInput({ ...interventionInput, text: e.target.value })}
                            onKeyDown={(e) => e.key === 'Enter' && handleAddIntervention(item.id!)}
                          />
                          <button 
                            onClick={() => handleAddIntervention(item.id!)}
                            className="p-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
                          >
                            <Check className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <button 
                          onClick={() => setInterventionInput({ id: item.id!, text: "" })}
                          className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 text-sm font-medium transition-colors"
                        >
                          <Plus className="w-4 h-4" /> Ghi chú Phun thuốc
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
