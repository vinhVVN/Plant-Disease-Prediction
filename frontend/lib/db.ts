import { openDB, DBSchema, IDBPDatabase } from 'idb';

// Kích thước tối đa của ảnh sau khi nén
const MAX_IMAGE_WIDTH = 800;
const JPEG_QUALITY = 0.8;

export interface PlantRecord {
  id: string;
  name: string;
  createdAt: number;
}

export interface PredictionRecord {
  id?: number;
  plantId: string;          // Foreign key liên kết với PlantRecord
  imageThumbnail: Blob;     // Ảnh thumbnail nén
  predictedClass: string;
  severityPercentage: number;
  createdAt: number;
  intervention: string | null;
}

interface AgroVisionDB extends DBSchema {
  plants: {
    key: string;
    value: PlantRecord;
    indexes: { 'by-date': number };
  };
  predictions: {
    key: number;
    value: PredictionRecord;
    indexes: { 
      'by-date': number;
      'by-plant': string;
    };
  };
}

let dbPromise: Promise<IDBPDatabase<AgroVisionDB>> | null = null;

if (typeof window !== 'undefined') {
  dbPromise = openDB<AgroVisionDB>('agrovision_db', 3, {
    upgrade(db, oldVersion, newVersion, transaction) {
      if (!db.objectStoreNames.contains('plants')) {
        const plantStore = db.createObjectStore('plants', { keyPath: 'id' });
        plantStore.createIndex('by-date', 'createdAt');
      }
      
      if (!db.objectStoreNames.contains('predictions')) {
        const predStore = db.createObjectStore('predictions', {
          keyPath: 'id',
          autoIncrement: true,
        });
        predStore.createIndex('by-date', 'createdAt');
        predStore.createIndex('by-plant', 'plantId');
      } else {
        // Nếu đã có bảng predictions từ version cũ, ta tạo thêm index by-plant
        const predStore = transaction.objectStore('predictions');
        if (!predStore.indexNames.contains('by-plant')) {
          predStore.createIndex('by-plant', 'plantId');
        }
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
 * Tạo một hồ sơ cây trồng mới
 */
export async function createPlant(name: string): Promise<string> {
  if (!dbPromise) throw new Error("DB not initialized");
  const db = await dbPromise;
  
  const id = Math.random().toString(36).substring(2, 15);
  const record: PlantRecord = {
    id,
    name,
    createdAt: Date.now(),
  };
  
  await db.add('plants', record);
  return id;
}

/**
 * Lấy danh sách tất cả hồ sơ cây trồng
 */
export async function getAllPlants(): Promise<PlantRecord[]> {
  if (!dbPromise) return [];
  const db = await dbPromise;
  
  const tx = db.transaction('plants', 'readonly');
  const store = tx.objectStore('plants');
  const index = store.index('by-date');
  
  const cursor = await index.openCursor(null, 'prev');
  const results: PlantRecord[] = [];
  
  let current = cursor;
  while (current) {
    results.push(current.value);
    current = await current.continue();
  }
  
  return results;
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
 * Lấy lịch sử chẩn đoán (có thể lọc theo plantId)
 */
export async function getAllPredictions(plantId?: string, limit: number = 100): Promise<PredictionRecord[]> {
  if (!dbPromise) return [];
  const db = await dbPromise;
  
  const tx = db.transaction('predictions', 'readonly');
  const store = tx.objectStore('predictions');
  
  let cursor;
  if (plantId) {
    // Nếu có plantId, truy vấn qua index by-plant
    const index = store.index('by-plant');
    cursor = await index.openCursor(IDBKeyRange.only(plantId));
  } else {
    const index = store.index('by-date');
    cursor = await index.openCursor(null, 'prev');
  }
  
  const results: PredictionRecord[] = [];
  
  let current = cursor;
  while (current && results.length < limit) {
    results.push(current.value);
    current = await current.continue();
  }
  
  // Sắp xếp lại theo createdAt tăng dần cho biểu đồ (Cũ -> Mới)
  results.sort((a, b) => a.createdAt - b.createdAt);
  
  return results;
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
