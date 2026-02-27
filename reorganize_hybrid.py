"""
HYBRID Dataset Reorganization - BEST APPROACH
Gabungkan Original + Augmented dengan proporsi yang seimbang

Target: 90%+ accuracy dengan data yang berkualitas
"""

import os
import shutil
from pathlib import Path
import random


def reorganize_hybrid():
    """
    Gabungkan Original + Augmented dengan sampling yang smart
    - Original: 100% (untuk kualitas)
    - Augmented: Sampling untuk balance classes
    """
    
    print("=" * 80)
    print("CREATING HYBRID DATASET (Original + Augmented)")
    print("=" * 80)
    print()
    
    base_dir = Path(__file__).parent
    source_dir = base_dir / "RLD_Dataset"
    target_dir = base_dir / "RLD_Dataset_Hybrid"
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    if not source_dir.exists():
        print(f"❌ Error: Source not found")
        return None
    
    # Mapping: (folder, original_path, augmented_path, target)
    mappings = [
        ("Bacterial Leaf Blight", "orginal", "augmented/Aug New Folder", "bacterial_blight"),
        ("Healthy _leaf", "orginal/leaf", "aug", "healthy"),
        ("Rice Blast", "orginal/Rice Blast", "augmented/Aug Rice Blast", "rice_blast"),
        ("Tungro", "orginal", "augmented/Aug Tungro", "tungro"),
    ]
    
    target_dir.mkdir(exist_ok=True)
    print(f"✓ Created: {target_dir}\n")
    
    total_images = 0
    class_counts = {}
    
    # Step 1: Copy ALL original images first
    print("STEP 1: Copying ALL ORIGINAL images...")
    for disease_folder, original_path, aug_path, target_name in mappings:
        print(f"\nProcessing: {disease_folder} (ORIGINAL)...")
        
        new_folder = target_dir / target_name
        new_folder.mkdir(exist_ok=True)
        
        source_folder = source_dir / disease_folder / original_path
        
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
        
        class_counts[target_name] = count
        total_images += count
        print(f"  ✓ {target_name:20s}: {count:5,d} original images")
    
    # Step 2: Add augmented images to balance classes
    print("\n" + "="*80)
    print("STEP 2: Adding AUGMENTED images for balance...")
    print("="*80)
    
    # Find max class count
    max_count = max(class_counts.values())
    target_per_class = int(max_count * 1.5)  # Target 1.5x dari class terbesar
    
    print(f"\nTarget per class: {target_per_class:,} images\n")
    
    for disease_folder, original_path, aug_path, target_name in mappings:
        current_count = class_counts[target_name]
        needed = target_per_class - current_count
        
        if needed <= 0:
            print(f"{target_name:20s}: Already sufficient ({current_count:,} images)")
            continue
        
        print(f"{target_name:20s}: Need {needed:,} more images...")
        
        new_folder = target_dir / target_name
        source_folder = source_dir / disease_folder / aug_path
        
        if not source_folder.exists():
            print(f"  ⚠ Augmented folder not found: {source_folder}")
            continue
        
        # Get all augmented files
        aug_files = [f for f in source_folder.iterdir() 
                     if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Random sample
        if len(aug_files) <= needed:
            # Use all if not enough
            selected = aug_files
        else:
            # Random sample
            selected = random.sample(aug_files, needed)
        
        added = 0
        for file in selected:
            dst = new_folder / f"aug_{file.name}"
            try:
                shutil.copy2(file, dst)
                added += 1
            except Exception as e:
                print(f"    Error: {file.name} - {e}")
        
        class_counts[target_name] += added
        total_images += added
        print(f"  ✓ Added {added:,d} augmented images")
    
    # Summary
    print("\n" + "="*80)
    print("HYBRID DATASET COMPLETE!")
    print("="*80)
    print(f"Total images: {total_images:,}")
    print(f"Location: {target_dir}\n")
    
    print("📊 Final Distribution:")
    for class_name, count in class_counts.items():
        percentage = count / total_images * 100 if total_images > 0 else 0
        print(f"  {class_name:20s}: {count:5,d} images ({percentage:5.2f}%)")
    
    print("\n" + "="*80)
    print("✅ READY FOR TRAINING!")
    print("="*80)
    print(f"\nTraining command:")
    print(f'  python train_model.py --dataset "{target_dir}"\n')
    
    return target_dir, total_images


if __name__ == "__main__":
    try:
        random.seed(42)  # For reproducibility
        reorganize_hybrid()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()