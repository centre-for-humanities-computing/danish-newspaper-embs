from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from datasets import Dataset
import numpy as np

app = typer.Typer()

def mean_pooling(dataset: Dataset):
    """
    Compute mean-pooled embeddings for each article in the dataset.
    """
    out = []
    for article in tqdm(dataset, desc="Pooling embeddings"):
        chunk_embs = article["embedding"]
        # Compute the mean across the chunk embeddings (assumes each article has a list of vectors)
        emb = np.mean(chunk_embs, axis=0)
        out.append(emb)
    return out

@app.command()
def main(
    input_ds: Path = typer.Argument(
        ...,
        help="Path to the dataset directory (saved with Dataset.save_to_disk)"
    ),
    output_ds: Path = typer.Argument(
        ...,
        help="Directory where the processed dataset will be saved"
    )
):
    """
    This script loads a dataset saved with Dataset.save_to_disk,
    applies mean pooling on the chunk embeddings for each article,
    and then saves the processed dataset to the specified output directory.
    """
    # Load the dataset from the directory
    ds_chunks = Dataset.load_from_disk(str(input_ds))
    
    # Compute number of chunks for each article
    n_chunks = [len(chunks) for chunks in ds_chunks["chunk"]]
    
    # Compute mean-pooled embeddings for each article
    pooled_embs = mean_pooling(ds_chunks)
    
    # Create a new dataset with the 'article' field, mean-pooled embeddings, and original chunk counts
    ds_pooled = Dataset.from_dict({
        "article_id": ds_chunks["article_id"],
        "embedding": pooled_embs,
        "n_chunks_orig": n_chunks,
    })

    # Ensure the output directory exists
    output_ds.mkdir(parents=True, exist_ok=True)
    
    # Save the processed dataset to disk
    ds_pooled.save_to_disk(str(output_ds))
    print(f"Saved processed dataset to {output_ds}")

if __name__ == "__main__":
    app()
