"""
Rice Disease Dataset Reorganization Script - FINAL VERSION
Pilihan: Original (kecil ~2.5K) atau Augmented (besar ~8K+)

Usage:
    python reorganize_dataset.py                # Augmented (recommended)
    python reorganize_dataset.py --original     # Original only
    python reorganize_dataset.py --both         # Both versions
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def reorganize_original():
    """Original data (~2,500 samples) - untuk testing"""
    print("=" * 80)
    print("REORGANIZING ORIGINAL DATASET")
    print("=" * 80 + "\n")
    
    base_dir = Path(__file__).parent
    source_dir = base_dir / "RLD_Dataset"
    target_dir = base_dir / "RLD_Dataset_Original"
    
    mappings = [
        ("Bacterial Leaf Blight", "orginal", "bacterial_blight"),
        ("Healthy _leaf", "orginal/leaf", "healthy"),
        ("Rice Blast", "orginal/Rice Blast", "rice_blast"),
        ("Tungro", "orginal", "tungro"),
    ]
    
    return process_dataset(source_dir, target_dir, mappings, "ORIGINAL")


def reorganize_augmented():
    """Augmented data (~8,000+ samples) - RECOMMENDED"""
    print("=" * 80)
    print("REORGANIZING AUGMENTED DATASET (RECOMMENDED)")
    print("=" * 80 + "\n")
    
    base_dir = Path(__file__).parent
    source_dir = base_dir / "RLD_Dataset"
    target_dir = base_dir / "RLD_Dataset_Augmented"
    
    mappings = [
        ("Bacterial Leaf Blight", "augmented/Aug New Folder", "bacterial_blight"),
        ("Healthy _leaf", "aug", "healthy"),
        ("Rice Blast", "augmented/Aug Rice Blast", "rice_blast"),
        ("Tungro", "augmented/Aug Tungro", "tungro"),
    ]
    
    return process_dataset(source_dir, target_dir, mappings, "AUGMENTED")


def process_dataset(source_dir, target_dir, mappings, dataset_type):
    """Process dataset reorganization"""
    
    if not source_dir.exists():
        print(f"❌ Error: Source not found: {source_dir}\n")
        return None
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}\n")
    print(f"📁 Using {dataset_type} data:")
    for disease, subpath, target in mappings:
        print(f"  {disease}/{subpath:35s} → {target}")
    print()
    
    target_dir.mkdir(exist_ok=True)
    total_images = 0
    
    for disease_folder, subfolder_path, target_name in mappings:
        print(f"Processing: {disease_folder}...")
        
        new_folder = target_dir / target_name
        new_folder.mkdir(exist_ok=True)
        
        source_folder = source_dir / disease_folder / subfolder_path
        
        if not source_folder.exists():
            print(f"  ⚠ Not found: {source_folder}")
            continue
        
        count = 0
        for file in source_folder.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    shutil.copy2(file, new_folder / file.name)
                    count += 1
                except Exception as e:
                    print(f"    Error: {file.name} - {e}")
        
        total_images += count
        print(f"  ✓ {target_name:20s}: {count:5,d} images")
    
    print(f"\n{'='*80}")
    print(f"{dataset_type} DATASET COMPLETE!")
    print(f"{'='*80}")
    print(f"Total images: {total_images:,}")
    print(f"Location: {target_dir}\n")
    
    return target_dir, total_images


def verify_dataset(target_dir):
    """Verify dataset distribution"""
    if not target_dir or not target_dir.exists():
        return
    
    print("📊 Dataset Distribution:")
    classes = ['bacterial_blight', 'rice_blast', 'tungro', 'healthy']
    
    total = 0
    for class_name in classes:
        folder = target_dir / class_name
        if folder.exists():
            count = sum(1 for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            total += count
    
    for class_name in classes:
        folder = target_dir / class_name
        if folder.exists():
            count = sum(1 for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {class_name:20s}: {count:5,d} images ({percentage:5.2f}%)")
    
    print(f"\n  {'TOTAL':20s}: {total:5,d} images\n")


def main():
    parser = argparse.ArgumentParser(description='Reorganize Rice Disease Dataset')
    parser.add_argument('--original', action='store_true', help='Original dataset (~2.5K)')
    parser.add_argument('--both', action='store_true', help='Both datasets')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🌾 RICE DISEASE DATASET REORGANIZATION")
    print("="*80 + "\n")
    
    results = []
    
    if args.both:
        print("📋 Mode: Creating BOTH datasets\n")
        r = reorganize_original()
        if r: results.append(('Original', r))
        print("\n" + "-"*80 + "\n")
        r = reorganize_augmented()
        if r: results.append(('Augmented', r))
    elif args.original:
        print("📋 Mode: Original dataset only\n")
        r = reorganize_original()
        if r: results.append(('Original', r))
    else:
        print("📋 Mode: Augmented dataset (RECOMMENDED)\n")
        r = reorganize_augmented()
        if r: results.append(('Augmented', r))
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        for name, (target_dir, total) in results:
            print(f"\n{name} Dataset:")
            verify_dataset(target_dir)
            print(f"Training command:")
            print(f'  python train_model.py --dataset "{target_dir}"')
        
        print("="*80)
        if len(results) > 1:
            print("\n💡 Use AUGMENTED for 90%+ accuracy!")
        print("\n✅ Completed!\n")
    else:
        print("\n❌ Failed. Check source directory.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)