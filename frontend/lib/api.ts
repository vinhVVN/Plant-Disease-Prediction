export interface CloudPredictionResult {
  predicted_class: string;
  confidence: number;
  uncertainty_warning: boolean;
  top5: { class: string; prob: number }[];
  inference_time_ms: number;
  recommendation: {
    disease_vi?: string;
    action: string;
    priority: string;
    description: string;
  };
  xai: {
    tta_b64_list: string[];
    pipeline_trace_b64: string;
    feature_map_b64: string;
    leaf_mask_b64: string;
    heatmap_b64: string;
    overlay_b64: string;
    severity: {
      percentage: number;
      level: string;
    }
  };
}

export async function runCloudInference(imageFile: File, modelName: string = 'efficientnetb0'): Promise<CloudPredictionResult> {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model_name', modelName);

  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Cloud API Error: ${response.statusText}`);
  }

  const data: CloudPredictionResult = await response.json();
  return data;
}
