
# %%
import os
import time
import logging
import warnings
from datasets import load_dataset

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from sklearn.utils import resample

# suppress only FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    filename='logs/classification_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True
)
timestamp = time.strftime("%Y%m%d-%H%M")
today_str = time.strftime("%Y%m%d")

if not os.path.exists('logs'):
    os.makedirs('logs')

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

# and we plan on ungrouping the series over train/test
# this is so stories of the same series are treated as the same group for classification, and not split over train/test
# remove suffixes (_a, _b, _c) from feuilleton_id
df['feuilleton_id'] = df['feuilleton_id'].str.replace(r'_[a-z]$', '', regex=True)

# first we give dummyIDs to the ones missing IDs
missing_mask = df['feuilleton_id'].isna()
df.loc[missing_mask, 'feuilleton_id'] = [f"noid_{i}" for i in range(missing_mask.sum())] # fill missing IDs with dummy IDs
# log this
print(f"After assigning dummy IDs, {df['feuilleton_id'].isna().sum()} missing feuilleton_ids remain.")
logging.info(f"After assigning dummy IDs, {df['feuilleton_id'].isna().sum()} missing feuilleton_ids remain.")
print(df['serialized'].value_counts())

# make tfidf as a column
tfidf = TfidfVectorizer(max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf'] = list(tfidf_matrix.toarray())

# inspect
df.head()

# %%

### UTILS FOR CLASSIFICATION ###

def evaluate_classifier(df, label, feature, n_splits=5, model_name="Logistic Regression"):
    """
    Evaluate a classifier using StratifiedGroupKFold to keep feuilleton_id grouped.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [labelcol] and precomputed features.
    labelcol : str
        Name of the label column.
    n_splits : int
        Number of folds.
    model_name : str
        Name of the classifier.
    Returns
    -------
    dict : Averaged precision, recall, f1, and accuracy per category.
    """
    logging.info(f"Starting evaluation with model: {feature}, label column: {label}, n_splits: {n_splits}, model_name: {model_name}")
    labels = df[label].unique()
    logging.info(f"Labels found: {labels}, counts: {df[label].value_counts().to_dict()}")
    metrics = {f"{label}_precision": [] for label in labels}
    metrics.update({f"{label}_recall": [] for label in labels})
    metrics.update({f"{label}_f1": [] for label in labels})
    metrics['accuracy'] = []

    # Features
    X = np.vstack(df[feature].values)
    logging.info(f"Using embeddings from {feature} with shape {X.shape}")

    # Labels & groups
    y = df[label].values
    groups = df['feuilleton_id'].values

    # Cross-validation
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    with tqdm(total=n_splits, desc="Cross-validation") as pbar:
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):        
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=fold)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            for label in labels:
                if label in report:
                    metrics[f'{label}_precision'].append(report[label]['precision'])
                    metrics[f'{label}_recall'].append(report[label]['recall'])
                    metrics[f'{label}_f1'].append(report[label]['f1-score'])
            metrics['accuracy'].append(report['accuracy'])

    # Average results
    averaged = {k: np.mean(v) for k, v in metrics.items()}
    averaged['model'] = model_name
    averaged['n_splits'] = n_splits

    return averaged

##
# checking that all embeddings are in the same format and not nan, and also that we have labels for all of them
def clean_embeddings(df, model):
    mask = df[model].apply(lambda x: isinstance(x, np.ndarray) and x.shape[0] > 1)
    print(f"Keeping {mask.sum()} out of {len(df)} samples for model {model}.")
    return df[mask].reset_index(drop=True)

##
# to balance classes, we will downsample the larger class to match the size of the smaller class, using sklearn's resample function
def balance_classes(df, label_col='label', random_state=42):
    # get all label variants
    labels = df[label_col].unique()
    # find the smallest class size
    min_size = df[label_col].value_counts().min()
    print(f'Classes: {labels}, min size: {min_size}')

    # downsample each class to match the smallest class
    balanced_dfs = []
    for label in labels:
        df_label = df[df[label_col] == label]
        if len(df_label) > min_size:
            df_label = resample(df_label, 
                                replace=False,  # no replacement
                                n_samples=min_size,
                                random_state=random_state)
        balanced_dfs.append(df_label)

    # concatenate all classes back together
    balanced_df = pd.concat(balanced_dfs)
    
    # shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logging.info(f"Balanced classes, {len(balanced_df)} samples. Counts: {balanced_df[label_col].value_counts().to_dict()}")
    print(balanced_df[label_col].value_counts())
    
    return balanced_df
    
##
# to prepare the df for classification, we will drop the metadata columns and keep only the label and the embeddings, and also balance the classes if needed
def prepare_df_for_classification(df, label, exclude_categories=None, balance=True):
    if exclude_categories:
        df = df[~df['subcategory_to_classify'].isin(exclude_categories)].reset_index(drop=True)
    # print the subcategory value counts
    if label == 'subcategory_to_classify':
        print(f"Value counts for subcategory:\n{df['subcategory_to_classify'].value_counts()}")
        logging.info(f"Value counts for subcategory:\n{df['subcategory_to_classify'].value_counts()}")
    # Drop metadata
    cols_to_drop = ['text', 'article_id','feuilleton_author','serialized'] + [x for x in ['label','subcategory_to_classify'] if x != label]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    # balance classes if needed
    if balance:
        df = balance_classes(df, label_col=label)
    return df

# %%
###### CLASSIFICATION ######

# SETUP
LABEL = 'label'  # 'label' or 'subcategory_to_classify'
BALANCE_CLASSES = True #True

# possible categories to exclude
exclude_cats = ['poem_fiction']#, 'anecdote_fiction']

######

# %%
# and embeddings
embeddings_df_classification = prepare_df_for_classification(df, label=LABEL, exclude_categories=exclude_cats, balance=BALANCE_CLASSES)

embedding_models = ['memo', 'oldnews', 'gemma', 'jina', 'bge', 'e-5', 'tfidf']
results_list = []

for model in embedding_models:
    embeddings_clean = clean_embeddings(embeddings_df_classification, model)
    results_emb = evaluate_classifier(embeddings_clean, feature=model, label=LABEL, n_splits=5)
    logging.info(f"Classification results for {model}:\n{results_emb}\n\n")
    # add modelname to results
    results_emb['model'] = model
    results_list.append(results_emb)

results_emb_df = pd.DataFrame(results_list)
logging.info(f"All embedding classification results:\n{results_emb_df.to_string(index=False)}\n\n")

results_emb_df.head(7)
# %%

