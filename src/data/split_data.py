import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import create_dirs, set_seed

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
SPLIT_DIR = BASE_DIR / "data" / "split"

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"
SEED = 42


def copy_files(files, labels, target_root):
    for file_path, label in zip(files, labels):
        class_dir = target_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, class_dir / file_path.name)


def split_dataset():
    set_seed(SEED)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    create_dirs(TRAIN_DIR, VAL_DIR, TEST_DIR)

    image_paths = []
    labels = []

    for class_dir in RAW_DIR.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(img_path)
                    labels.append(class_dir.name)

    if len(image_paths) == 0:
        raise ValueError("Không tìm thấy ảnh trong data/raw.")

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_labels
    )

    copy_files(train_files, train_labels, TRAIN_DIR)
    copy_files(val_files, val_labels, VAL_DIR)
    copy_files(test_files, test_labels, TEST_DIR)

    print("Chia dữ liệu hoàn tất")
    print(f"Train: {len(train_files)} ảnh")
    print(f"Val: {len(val_files)} ảnh")
    print(f"Test: {len(test_files)} ảnh")


if __name__ == "__main__":
    split_dataset()