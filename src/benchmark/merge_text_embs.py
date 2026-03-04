import pandas as pd
from datasets import Dataset
import os

texts = pd.read_csv('../data/test_task/sample_500.csv', sep='\t', index_col=0)
texts = texts.rename(columns={'id': 'article_id'})

embs_list = ['e-5', 'jina', 'memo', 'old', 'bge', 'gemma']
path_root = '../data/test_task/pooled/'

df_all_embs = texts.reset_index(drop=False)

for emb_name in embs_list:
    # Load embeddings dataset
    ds_path = os.path.join(path_root, emb_name)
    embs_ds = Dataset.load_from_disk(ds_path)
    embs_df = embs_ds.to_pandas()

    # Keep only id + embedding and rename
    embs_df = embs_df[['article_id', 'embedding']].rename(columns={'embedding': emb_name})

    # Merge directly on 'article_id'
    df_all_embs = df_all_embs.merge(embs_df, on='article_id', how='left')

# Convert back to HF Dataset and save
new_ds = Dataset.from_pandas(df_all_embs.reset_index(drop=True))
save_path = os.path.join(path_root, 'merged_all')
new_ds.save_to_disk(save_path)

print(f"Merged dataset saved to: {save_path}")