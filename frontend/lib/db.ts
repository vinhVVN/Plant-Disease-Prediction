import { openDB, DBSchema, IDBPDatabase } from 'idb';

// Kích thước tối đa của ảnh sau khi nén
const MAX_IMAGE_WIDTH = 800;
const JPEG_QUALITY = 0.8;

export interface PredictionRecord {
  id?: number;
  imageThumbnail: Blob;     // Ảnh thumbnail nén
  predictedClass: string;
  severityPercentage: number;
  createdAt: number;
  intervention: string | null;
}

interface AgroVisionDB extends DBSchema {
  predictions: {
    key: number;
    value: PredictionRecord;
    indexes: { 'by-date': number };
  };
}

let dbPromise: Promise<IDBPDatabase<AgroVisionDB>> | null = null;

if (typeof window !== 'undefined') {
  dbPromise = openDB<AgroVisionDB>('agrovision_db', 2, {
    upgrade(db, oldVersion, newVersion, transaction) {
      if (!db.objectStoreNames.contains('predictions')) {
        const store = db.createObjectStore('predictions', {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex('by-date', 'createdAt');
      }
    },
  });
}

/**
 * Nén ảnh bằng thẻ Canvas để tránh tràn bộ nhớ IndexedDB
 */
export async function compressImage(blob: Blob, maxWidth: number = MAX_IMAGE_WIDTH, quality: number = JPEG_QUALITY): Promise<Blob> {
  const img = new Image();
  const url = URL.createObjectURL(blob);
  
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = url;
  });

  let width = img.width;
  let height = img.height;

  // Tính toán tỷ lệ nếu ảnh quá lớn
  if (width > maxWidth) {
    height = Math.round((height * maxWidth) / width);
    width = maxWidth;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) throw new Error("Could not get 2d context");
  ctx.drawImage(img, 0, 0, width, height);
  URL.revokeObjectURL(url);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (compressedBlob) => {
        if (compressedBlob) resolve(compressedBlob);
        else reject(new Error("Canvas to Blob failed"));
      },
      'image/jpeg',
      quality
    );
  });
}

/**
 * Lưu lịch sử chẩn đoán vào IndexedDB
 */
export async function savePrediction(record: Omit<PredictionRecord, 'id' | 'createdAt'>) {
  if (!dbPromise) return;
  const db = await dbPromise;
  
  const finalRecord: PredictionRecord = {
    ...record,
    createdAt: Date.now(),
  };

  return db.add('predictions', finalRecord);
}

/**
 * Lấy lịch sử chẩn đoán
 */
export async function getAllPredictions(limit: number = 50): Promise<PredictionRecord[]> {
  if (!dbPromise) return [];
  const db = await dbPromise;
  
  const tx = db.transaction('predictions', 'readonly');
  const store = tx.objectStore('predictions');
  const index = store.index('by-date');
  
  // Lấy danh sách mới nhất
  const cursor = await index.openCursor(null, 'prev');
  const results: PredictionRecord[] = [];
  
  let current = cursor;
  while (current && results.length < limit) {
    results.push(current.value);
    current = await current.continue();
  }
  
  // Đảo ngược mảng để trả về theo thứ tự thời gian tăng dần cho biểu đồ (Cũ -> Mới)
  return results.reverse();
}

/**
 * Thêm ghi chú can thiệp (Phun thuốc) vào một record đã có
 */
export async function addIntervention(id: number, interventionNote: string) {
  if (!dbPromise) return;
  const db = await dbPromise;
  
  const tx = db.transaction('predictions', 'readwrite');
  const store = tx.objectStore('predictions');
  
  const record = await store.get(id);
  if (!record) throw new Error("Record not found");
  
  record.intervention = interventionNote;
  await store.put(record);
  await tx.done;
}
