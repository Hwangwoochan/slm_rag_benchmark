import os
from datasets import load_dataset

# 저장할 base 디렉토리
BASE_DIR = "./data"

# 우리가 쓰기로 한 configs
configs = ["delucionqa", "emanual", "techqa"]

# base 디렉토리 없으면 생성
os.makedirs(BASE_DIR, exist_ok=True)

datasets = {}

for cfg in configs:
    print(f"\nLoading {cfg}...")
    ds = load_dataset("rungalileo/ragbench", cfg)
    datasets[cfg] = ds

    # config별 디렉토리 생성
    cfg_dir = os.path.join(BASE_DIR, cfg)
    os.makedirs(cfg_dir, exist_ok=True)

    # split별 저장
    for split in ds.keys():
        save_path = os.path.join(cfg_dir, f"{split}.json")
        print(f"Saving {cfg}/{split} → {save_path}")
        ds[split].to_json(save_path)

print("\nAll datasets saved successfully.")
