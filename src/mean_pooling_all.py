from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from datasets import Dataset
import numpy as np

app = typer.Typer()

def mean_pooling(dataset: Dataset):
    out = []
    for article in tqdm(dataset, desc="Pooling embeddings"):
        chunk_embs = article["embedding"]
        if not chunk_embs:  # empty list
            emb = None
        else:
            emb = np.mean(chunk_embs, axis=0).tolist()  # convert to Python list
        out.append(emb)
    return out

def mean_pooling_fast(example):
    chunk_embs = example["embedding"]
    if not chunk_embs:
        return {"pooled": None}
    return {"pooled": np.mean(chunk_embs, axis=0).tolist()}

default_output_dir = Path(__file__).parent /".." / "data" /"all"

@app.command()
def main(
    input_ds: Path = typer.Argument(
        'embeddings_old_news',
        help="Path to the dataset directory (saved with Dataset.save_to_disk)"
    ),
    output_dir: Path = typer.Option(default_output_dir, help="Directory to save embeddings"),
):
    """
    This script loads a dataset saved with Dataset.save_to_disk,
    applies mean pooling on the chunk embeddings for each article,
    and then saves the processed dataset to the specified output directory.
    """
    # Load the dataset from the directory
    ds_chunks = Dataset.load_from_disk(default_output_dir / input_ds)
    
    # Compute mean-pooled embeddings for each article
    #pooled_embs = mean_pooling(ds_chunks)

    #ds_pooled = ds_chunks.add_column('pooled', column=pooled_embs)

    ds_pooled = ds_chunks.map(mean_pooling_fast, desc="Pooling", num_proc=4)
    
    # Save the processed dataset to disk
    ds_pooled.save_to_disk(output_dir / "pooled_old_news")
    print(f"Saved processed dataset to {output_dir} / pooled_old_news")

if __name__ == "__main__":
    app()