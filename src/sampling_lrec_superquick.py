from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np

DATASET_NAME = "chcaa/eno-embs-old-news"
OUTPUT_PATH = "../../../ROOT/DATA/ENO_embs_251001/sample_2000_each"
SAMPLE_SIZE = 2000
RANDOM_SEED = 42

print(f"📂 Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"✅ Loaded {len(dataset):,} rows.\n")

# Build an index of row positions per newspaper (one fast scan)
print("🗂️  Building index of rows per newspaper...")
indices_by_nsp = {}
for i, nsp in enumerate(tqdm(dataset["newspaper"], desc="Indexing")):
    indices_by_nsp.setdefault(nsp, []).append(i)

# Sample from each group efficiently
print(f"\n🎲 Sampling up to {SAMPLE_SIZE} articles per newspaper...")
np.random.seed(RANDOM_SEED)

sampled_slices = []
for nsp, idxs in tqdm(indices_by_nsp.items(), desc="Sampling newspapers"):
    n = min(len(idxs), SAMPLE_SIZE)
    sampled_idx = np.random.choice(idxs, n, replace=False)
    sampled_slices.append(dataset.select(sampled_idx))

sampled_dataset = concatenate_datasets(sampled_slices)

print(f"\n✅ Sampled total: {len(sampled_dataset):,} rows across {len(indices_by_nsp)} newspapers.")

sampled_dataset.save_to_disk(OUTPUT_PATH)
print(f"💾 Saved sampled dataset to: {OUTPUT_PATH}")
