import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

def train_classifier(
    df,
    embedding_col: str = "old",
    test_size: float = 0.2,
    random_state: int = 42,
    n_samples_per_class: int = 308,
):
    """
    Train a multinomial logistic regression classifier on pooled embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `embedding_col` with embeddings and `category_gold` as labels.
    embedding_col : str
        Column name with embedding vectors (e.g. 'old').
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    n_samples_per_class : int
        Max number of samples per class (for balancing).

    Returns
    -------
    clf : LogisticRegression
        Trained classifier.
    report : dict
        Classification report on held-out test set.
    acc : float
        Accuracy score on test set.
    """
    # Balance dataset by resampling per class
    balanced_df = (
        df.groupby("category_gold", group_keys=False)
        .apply(
            lambda x: resample(
                x,
                n_samples=min(len(x), n_samples_per_class),
                random_state=random_state,
            )
        )
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # Features + labels
    X = np.vstack(balanced_df[embedding_col].values)
    y = balanced_df["category_gold"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train multinomial logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    acc = accuracy_score(y_test, y_pred)

    return clf, report, acc


def make_predict_fn(clf, embedding_col="pooled"):
    """Closure that returns a batch prediction function using a fitted clf."""

    def add_predictions_batch(batch):
        X = np.array(batch[embedding_col], dtype=np.float32)
        preds = clf.predict(X)
        return {"predicted_category": preds}

    return add_predictions_batch


if __name__ == "__main__":

    # Paths
    gold_path = "../../data/test_task/subset_final_gold_sample.csv"
    embs_path = "../../data/test_task/pooled/merged_all"
    model_out = "../../models/logreg_classifier.joblib"

    # Load gold labels
    sample_gold = pd.read_csv(gold_path)

    # Load embeddings of sample
    merged_all = Dataset.load_from_disk(embs_path).to_pandas()

    # Merge gold labels with embeddings
    embs_gold_orig = sample_gold.merge(
        merged_all[["article_id", "old"]], on="article_id"
    )

    # Train classifier
    clf, report, acc = train_classifier(embs_gold_orig, embedding_col="old")
    print("✅ Training done.")
    print("Accuracy:", acc)

    # Save classifier to disk
    joblib.dump(clf, model_out)
    print(f"💾 Classifier saved to {model_out}")

    # Load target dataset for predictions
    dataset = load_dataset("chcaa/eno-embs-old-news", split="train")

    # Add predictions in batches
    dataset = dataset.map(
        make_predict_fn(clf, embedding_col="pooled"),
        batched=True,
        batch_size=1024,
    )
    print("✅ Predictions added to dataset.")

    # Push to Hugging Face Hub
    dataset.push_to_hub("chcaa/eno-embs-old-news")
    print("🚀 Dataset pushed to hub.")