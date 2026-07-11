import os
import shutil
import argparse
import random
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1,
                   test_ratio=0.1, seed=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Rasio train + val + test harus berjumlah 1.0"

    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")

    class_names = ['bacterial_blight', 'rice_blast', 'tungro', 'healthy']
    splits = ['train', 'val', 'test']

    # Buat struktur folder output
    for split in splits:
        for cls in class_names:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {split: {} for split in splits}

    print("=" * 80)
    print("MEMISAHKAN DATASET MENJADI TRAIN / VAL / TEST (FOLDER TERPISAH)")
    print("=" * 80)
    print(f"Sumber      : {source_dir}")
    print(f"Output      : {output_dir}")
    print(f"Rasio split : Train {train_ratio*100:.0f}% | "
          f"Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")
    print(f"Random seed : {seed}\n")

    for cls in class_names:
        class_dir = source_dir / cls
        if not class_dir.exists():
            print(f"⚠  Folder tidak ditemukan: {class_dir} — dilewati.\n")
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # sisa, hindari kehilangan akibat pembulatan

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        file_groups = {'train': train_files, 'val': val_files, 'test': test_files}

        for split, files in file_groups.items():
            for fname in files:
                src = class_dir / fname
                dst = output_dir / split / cls / fname
                shutil.copy2(src, dst)
            summary[split][cls] = len(files)

        print(f"  {cls:20s}: total={n_total:4d}  →  "
              f"train={n_train:4d}, val={n_val:4d}, test={n_test:4d}")

    print("\n" + "=" * 80)
    print("RINGKASAN HASIL SPLIT")
    print("=" * 80)
    for split in splits:
        total_split = sum(summary[split].values())
        print(f"\n{split.upper()} — total {total_split} gambar:")
        for cls in class_names:
            count = summary[split].get(cls, 0)
            pct = (count / total_split * 100) if total_split > 0 else 0
            print(f"    {cls:20s}: {count:4d} ({pct:5.2f}%)")

    grand_total = sum(sum(summary[s].values()) for s in splits)
    print(f"\n{'TOTAL KESELURUHAN':20s}: {grand_total}")
    print(f"\n✅ Selesai. Dataset tersimpan di folder terpisah:")
    print(f"   {output_dir}/train/<kelas>/  ({sum(summary['train'].values())} gambar)")
    print(f"   {output_dir}/val/<kelas>/    ({sum(summary['val'].values())} gambar)")
    print(f"   {output_dir}/test/<kelas>/   ({sum(summary['test'].values())} gambar)")
    print("\nNOTE: Augmentasi kelas minoritas TIDAK dilakukan di sini.")
    print("      Augmentasi hanya akan diterapkan pada folder train/ saat")
    print("      training (lihat train_modelrf801010.py), agar val/ dan test/")
    print("      tetap murni data asli tanpa kebocoran (data leakage).\n")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset menjadi folder train/val/test terpisah secara fisik'
    )
    parser.add_argument('--source', type=str, required=True,
                         help='Folder sumber berisi subfolder per kelas '
                              '(bacterial_blight, rice_blast, tungro, healthy). '
                              'Gunakan dataset ASLI (bukan yang sudah diaugmentasi).')
    parser.add_argument('--output', type=str, default='RLD_Dataset_Split',
                         help='Folder output hasil split (default: RLD_Dataset_Split)')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()