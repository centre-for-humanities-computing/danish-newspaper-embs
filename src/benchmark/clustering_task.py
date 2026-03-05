
# %%
import numpy as np
import pandas as pd
from datasets import load_dataset
import os

from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import logging
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# %%

# Configure logging
logging.basicConfig(
    filename='logs/clustering_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True
)
if not os.path.exists('logs'):
    os.makedirs('logs')

timestamp = time.strftime("%Y%m%d-%H%M")
today_str = time.strftime("%Y%m%d")

# testset path
testset_path = "chcaa/fiction-nonifction-testset-newspaper-embs"

# %%

# Load HF dataset
dataset = load_dataset(testset_path, split="train")
df = dataset.to_pandas()
print(f"Loaded dataset with {len(df)} samples.")

# make essay category nan
df['subcategory'] = df['subcategory'].replace({'essay': np.nan})

# just adding some subcats in subcategory col, so where label is fiction, make it literary; where non-fiction, make it news
df['subcategory'] = df.apply(lambda row: 'literary' if pd.isna(row['subcategory']) and row['label'] == 'fiction' else ('news' if pd.isna(row['subcategory']) and row['label'] == 'non-fiction' else row['subcategory']), axis=1)

# add label to subcategory overall, so bio + _ + fiction, etc
df['subcategory_to_classify'] = df.apply(lambda row: f"{row['subcategory']}_{row['label']}" if pd.notna(row['subcategory']) else row['subcategory'], axis=1)

# drop subcategory
df = df.drop(columns=['subcategory'])
print(df['subcategory_to_classify'].value_counts())

# # filter out poetry_fiction and anecdote_fiction as described in paper
exclude_cats = ['poem_fiction', 'anecdote_fiction']
df = df[~df['subcategory_to_classify'].isin(exclude_cats)].reset_index(drop=True)
print(f"After filtering out {exclude_cats}, dataset has {len(df)} samples.")

df.head()

# %%

# remove rows without feuilleton_id, i.e., we are only clustering those that can be clustered:)
only_id = df[~df['feuilleton_id'].isna()]
# remove suffixes (_a, _b, _c) from feuilleton_id
only_id['feuilleton_id'] = only_id['feuilleton_id'].str.replace(r'_[a-z]$', '', regex=True)
print("Number of unique feuilleton_ids:", only_id['feuilleton_id'].nunique())

# do we want to use only fiction?
#only_id = only_id[only_id['label'] == 'fiction'].reset_index(drop=True)
only_id.head()

# %%
# add tfidf for each embedding
tfidf = TfidfVectorizer(max_features=10000)
tfidf_matrix = tfidf.fit_transform(only_id['text'])
# add tfidf as a new column
only_id['tfidf'] = list(tfidf_matrix.toarray())

only_id.head()
# %% 

# -- Clustering task --
def clean_embeddings(df, model):
    mask = df[model].apply(lambda x: isinstance(x, np.ndarray) and x.shape[0] > 1)
    print(f"Keeping {mask.sum()} out of {len(df)} samples for model {model}.")
    return df[mask].reset_index(drop=True)

save_dict = {}

emd_cols = ['memo', 'oldnews', 'gemma', 'jina', 'bge','e-5', 'tfidf']

for embedding in emd_cols:
    print(f"Clustering for {embedding}")
    # remove faulty embeddings
    #tmp = clusters[clusters[embedding].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
    tmp = clean_embeddings(only_id, embedding)
    print(f"Number of rows in {embedding} after filtering: {len(tmp)}")
    logging.info(f"Number of rows in {embedding} after filtering: {len(tmp)}")

    # get the clusters
    X = np.vstack(tmp[embedding].values)
    y = tmp["feuilleton_id"].values
    # Set number of clusters to number of unique feuilleton_ids
    n_clusters = np.unique(y).shape[0]
    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # note down the number of clusters
    print("number of clusters:", len(np.unique(clusters)), ", should be same as:", len(np.unique(tmp["feuilleton_id"].values)))

    # get performance metrics
    ari = round(adjusted_rand_score(y, clusters),3)
    print("Adjusted Rand Index:", ari)
    logging.info(f"Adjusted Rand Index: {ari}")

    # get v-score
    v_score = round(v_measure_score(y, clusters),3)
    print("V-measure Score:", v_score)
    logging.info(f"V-measure Score: {v_score}")

    # save 
    save_dict[embedding] = {
        "ari": ari,
        "v_score": v_score,
        "n_clusters": n_clusters
    }

# write savedict to file
with open("logs/clustering_report.txt", "a") as f:
    f.write("\n\n")
    f.write("Clustering report:\n")
    for path, metrics in save_dict.items():
        f.write(f"{path}: {metrics}\n")
    f.write("\n\n")

# %%