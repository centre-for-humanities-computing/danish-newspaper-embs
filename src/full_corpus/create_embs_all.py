#!pip install einops for jina

import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq


app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]

def find_max_tokens(tokenizer):
    max_length = tokenizer.model_max_length
    if max_length > 9000:
        max_length = 510
    try:
        tokenizer("This is a test sentence.")
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        max_length = 510
    return max_length

default_output_dir = Path(__file__).parent /"../.." / "data" /"all"

@app.command()
def main(
    output_dir: Path = typer.Option(default_output_dir, help="Directory to save embeddings"),
    model_name: str = typer.Option("JohanHeinsen/Old_News_Segmentation_SBERT_V0.1", help="SentenceTransformer model"),
    batch_size: int = typer.Option(512, help="Batch size for each GPU process"),
    max_articles: int = typer.Option(None, help="Limit total number of articles to process (for testing)"),
):
    """
    Scalable embedding pipeline:
    - Streams dataset from Hugging Face
    - Preprocesses texts into chunks
    - Encodes with a persistent multi-process GPU pool
    - Saves results incrementally to Parquet
    """

    model = SentenceTransformer(model_name, trust_remote_code=True)
    max_tokens = find_max_tokens(model.tokenizer)
    logger.info(f"Max tokens for {model_name}: {max_tokens}")

    writer = None

    ds = load_from_disk(default_output_dir /"preprocessed_512")

# Start multi-process pool (workers pinned to GPUs)
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    idx_texts = [(i, t) for i, chunk in enumerate(ds["chunks"]) for t in chunk]
    idx, texts = zip(*idx_texts)

    print("starting to encode")    
    embs = model.encode(
        texts,
        pool=pool,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    print("done encoding")

    embs_col = [list() for i in range(len(ds))]
    for idx, emb in zip(idx, embs):
        embs_col[idx].append(
        emb.tolist() if hasattr(emb, "tolist") else emb
    )

    print("done mapping embeddings to ids")
    ds = ds.add_column(name="embedding", column=embs_col)

    def encode_row(example):
        chunks = example["chunks"]
        if not chunks:
            return {"embedding": []}
        embs = model.encode(chunks, pool=pool, batch_size=batch_size, convert_to_numpy=True)
        return {"embedding": [emb.tolist() for emb in embs]}

    #ds = ds.map(encode_row, batched=False)


    print("saving to disk - you did it!")
    # Save embeddings dataset
    ds.save_to_disk(output_dir / "embeddings_old_news")

    # Stop pool
    model.stop_multi_process_pool(pool)

    logger.info(f"✅ Finished. Saved embeddings to {output_dir/'embeddings_old_news'}")

if __name__ == "__main__":
    app()
