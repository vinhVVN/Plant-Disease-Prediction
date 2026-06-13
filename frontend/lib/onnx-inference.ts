import * as ort from 'onnxruntime-web';
import CLASS_NAMES from '../../shared/class_names.json';

// Cấu hình đường dẫn tới file WASM của onnxruntime-web.
// Trong package.json, script 'postinstall' sẽ copy các file .wasm từ node_modules ra public/ort-wasm/
ort.env.wasm.wasmPaths = '/ort-wasm/';

// ImageNet Mean và Std dùng chung cho cả PyTorch và Frontend
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const IMAGE_SIZE = 224;

export interface InferenceResult {
  className: string;
  confidence: number;
}

/**
 * Tiền xử lý ảnh (Resize -> Cắt trung tâm -> Chuẩn hóa ImageNet)
 */
async function preprocessImage(imageBlob: Blob): Promise<Float32Array> {
  const img = new Image();
  const imageUrl = URL.createObjectURL(imageBlob);

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = imageUrl;
  });

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Không thể khởi tạo Canvas 2D context');

  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;

  // Tính toán để cắt ảnh vuông (center crop)
  const scale = Math.max(IMAGE_SIZE / img.width, IMAGE_SIZE / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  const x = (IMAGE_SIZE - w) / 2;
  const y = (IMAGE_SIZE - h) / 2;

  // Vẽ ảnh đã resize và crop vào canvas
  ctx.drawImage(img, x, y, w, h);
  URL.revokeObjectURL(imageUrl);

  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;

  // Float32Array chứa dữ liệu ảnh dưới dạng [1, 3, 224, 224] (Batch, Channel, Height, Width)
  const tensorData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);

  // Normalize theo công thức: (pixel/255 - mean) / std
  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const r = imageData[i * 4 + 0] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    // Kênh R
    tensorData[i] = (r - MEAN[0]) / STD[0];
    // Kênh G (offset = SIZE * SIZE)
    tensorData[i + IMAGE_SIZE * IMAGE_SIZE] = (g - MEAN[1]) / STD[1];
    // Kênh B (offset = 2 * SIZE * SIZE)
    tensorData[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = (b - MEAN[2]) / STD[2];
  }

  return tensorData;
}

/**
 * Hàm Softmax để chuyển mảng logits thành xác suất (probabilities)
 */
function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sumExps);
}

/**
 * Chạy suy luận (Inference) bằng ONNX Runtime Web
 */
export async function runEdgeInference(imageBlob: Blob): Promise<InferenceResult[]> {
  try {
    // 1. Tiền xử lý ảnh
    const inputData = await preprocessImage(imageBlob);

    // 2. Tạo Tensor ONNX
    const tensor = new ort.Tensor('float32', inputData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);

    // 3. Khởi tạo session (chỉ dùng wasm backend để đảm bảo chạy trên mọi thiết bị)
    const session = await ort.InferenceSession.create('/models/best_model.onnx', {
      executionProviders: ['wasm']
    });

    // 4. Chạy model
    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = tensor;
    
    const results = await session.run(feeds);
    
    // 5. Trích xuất Output
    const outputTensor = results[session.outputNames[0]];
    const logits = Array.from(outputTensor.data as Float32Array);
    
    // 6. Tính toán Softmax và lấy Top 5
    const probabilities = softmax(logits);
    
    const predictions: InferenceResult[] = probabilities
      .map((prob, index) => ({
        className: CLASS_NAMES[index],
        confidence: prob,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5); // Lấy Top 5

    return predictions;
  } catch (error) {
    console.error("Lỗi khi chạy ONNX Inference:", error);
    throw error;
  }
}
