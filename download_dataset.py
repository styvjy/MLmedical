"""
Dataset Download and Preparation Utility for Chest X-Ray Pneumonia Classifier.
Provides automated download via KaggleHub/Kaggle CLI, dataset verification,
and automatic restructuring of nested folders.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

from src.config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
from src.data_loader import verify_dataset_structure

def reorganize_nested_folders(target_dir: Path) -> bool:
    """
    Detects and reorganizes common zip extraction patterns like:
    dataset/chest_xray/chest_xray/train -> dataset/train
    dataset/chest_xray/train -> dataset/train
    """
    target_dir = Path(target_dir)
    nested_candidates = [
        target_dir / "chest_xray" / "chest_xray",
        target_dir / "chest_xray",
        target_dir / "ChestXRay2017" / "chest_xray"
    ]

    for candidate in nested_candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
            print(f"[INFO] Found nested dataset directory at: {candidate}")
            for split in ["train", "val", "test"]:
                src_split = candidate / split
                dst_split = target_dir / split
                if src_split.exists() and not dst_split.exists():
                    print(f"  Moving {src_split} -> {dst_split}...")
                    shutil.move(str(src_split), str(dst_split))
            return True
    return False

def download_with_kagglehub(target_dir: Path) -> bool:
    """Attempts automated download using the modern kagglehub package."""
    try:
        import kagglehub
        print("[INFO] Attempting download via KaggleHub ('paultimothymooney/chest-xray-pneumonia')...")
        downloaded_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"[INFO] KaggleHub downloaded files to: {downloaded_path}")
        
        # Link or copy to target_dir
        src_path = Path(downloaded_path)
        # Check for nested chest_xray folder
        if (src_path / "chest_xray").exists():
            src_path = src_path / "chest_xray"

        for split in ["train", "val", "test"]:
            split_src = src_path / split
            split_dst = target_dir / split
            if split_src.exists() and not split_dst.exists():
                print(f"Copying {split} split to {split_dst}...")
                shutil.copytree(str(split_src), str(split_dst))

        return True
    except ImportError:
        print("[INFO] kagglehub is not installed. Run 'pip install kagglehub' for one-command download.")
        return False
    except Exception as e:
        print(f"[WARNING] kagglehub download failed: {e}")
        return False

def print_manual_instructions():
    print("\n" + "=" * 70)
    print(" MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("1. Visit the Kaggle Chest X-Ray Images (Pneumonia) page:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Click 'Download' (archive is ~1.15 GB).")
    print("3. Extract the contents so that your folder structure looks like this:")
    print("   c:\\satya project AI\\dataset\\")
    print("   ? train\\")
    print("   ?   ? NORMAL\\       (contains .jpeg images)")
    print("   ?   ?? PNEUMONIA\\    (contains .jpeg images)")
    print("   ? val\\")
    print("   ?   ? NORMAL\\")
    print("   ?   ?? PNEUMONIA\\")
    print("   ? test\\")
    print("       ? NORMAL\\")
    print("       ?? PNEUMONIA\\")
    print("")
    print("Alternative: Using Kaggle CLI (if you have kaggle.json configured):")
    print("   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p dataset --unzip")
    print("   python download_dataset.py --reorganize")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Dataset Download and Verification Helper")
    parser.add_argument("--reorganize", action="store_true", help="Reorganize extracted nested folders")
    parser.add_argument("--auto", action="store_true", help="Attempt automated download via kagglehub")
    args = parser.parse_args()

    print("=" * 60)
    print(" Chest X-Ray Pneumonia Dataset Checker")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if args.reorganize:
        reorganize_nested_folders(DATASET_DIR)

    is_valid, counts, msg = verify_dataset_structure(DATASET_DIR)
    if is_valid:
        print("\n[SUCCESS] Dataset is properly structured and ready for training!")
        for split, cls_counts in counts.items():
            print(f"  - {split.upper()}: {cls_counts['NORMAL']} Normal, {cls_counts['PNEUMONIA']} Pneumonia")
        return

    # Check if nested folder exists before failing
    if reorganize_nested_folders(DATASET_DIR):
        is_valid, counts, msg = verify_dataset_structure(DATASET_DIR)
        if is_valid:
            print("\n[SUCCESS] Successfully reorganized nested dataset!")
            return

    if args.auto:
        print("[INFO] Attempting automated download...")
        if download_with_kagglehub(DATASET_DIR):
            is_valid, counts, msg = verify_dataset_structure(DATASET_DIR)
            if is_valid:
                print("\n[SUCCESS] Download completed and dataset verified!")
                return

    print(f"\n[STATUS] {msg}")
    print_manual_instructions()

if __name__ == "__main__":
    main()
