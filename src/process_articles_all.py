import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq


app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def clean_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])\s+", r"\1 ", text)
    return text.strip()


def simple_sentencize(text: str) -> list[str]:
    sentences = re.findall(r"[^.!?]*[.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list[str], max_tokens: int, tokenizer) -> list[str]:
    output, current_chunk, chunk_len = [], [], 0
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        seq_len = len(tokens["input_ids"])
        if chunk_len + seq_len > max_tokens:
            if len(current_chunk) == 0:
                output.extend(split_long_sentence(sentence, max_tokens, tokenizer))
            else:
                output.append(" ".join(current_chunk))
                current_chunk, chunk_len = [], 0
        current_chunk.append(sentence)
        chunk_len += seq_len
    if current_chunk:
        output.append(" ".join(current_chunk))
    return output


def split_long_sentence(sentence: str, max_tokens: int, tokenizer) -> list[str]:
    words, parts, current_part, current_len = sentence.split(), [], [], 0
    for word in words:
        tokens = tokenizer.tokenize(word)
        seq_len = len(tokens["input_ids"])
        if current_len + seq_len > max_tokens:
            parts.append(" ".join(current_part))
            current_part, current_len = [], 0
        current_part.append(word)
        current_len += seq_len
    if current_part:
        parts.append(" ".join(current_part))
    return parts


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


def stream_batches(ds, batch_size):
    """Yield batches from a streaming dataset"""
    batch = []
    for row in ds:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def preprocess(example, model_name="jinaai/jina-embeddings-v3", max_tokens=8194, prefix=None):
    # initialize tokenizer locally in worker
    if not hasattr(preprocess, "tokenizer"):
        preprocess.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sentences = simple_sentencize(example["text"])
    chunks = []
    current_chunk, chunk_len = [], 0
    for sentence in sentences:
        tokens = preprocess.tokenizer(sentence)
        seq_len = len(tokens["input_ids"])
        if chunk_len + seq_len > max_tokens:
            if len(current_chunk) == 0:
                chunks.append(sentence)  # you could also split long sentences
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk, chunk_len = [], 0
        current_chunk.append(sentence)
        chunk_len += seq_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    example["chunks"] = [f"{prefix} {c}" if prefix else c for c in chunks]
    return example

default_output_dir = Path(__file__).parent /".." / "data" /"all"

@app.command()
def main(
    dataset_name: str = typer.Option("JohanHeinsen/ENO", help="Hugging Face dataset name"),
    split: str = typer.Option("train", help="Which split to use"),
    output_dir: Path = typer.Option(default_output_dir, help="Directory to save embeddings"),
    model_name: str = typer.Option("jinaai/jina-embeddings-v3", help="SentenceTransformer model"),
    newspapers: str = typer.Option(None, help="Comma-separated list of newspaper names to include"),
    prefix: str = typer.Option(None, help="Prefix for each chunk"),
    prefix_description: str = typer.Option(None, help="Used in output dir name"),
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

    # Build output path
    mname = model_name.replace("/", "__")
    if prefix:
        if prefix_description:
            output_path = output_dir / f"emb__{mname}_{prefix_description}"
        else:
            prefix_hash = hash_prompt(prefix)
            output_path = output_dir / f"emb__{mname}_{prefix_hash}"
            logger.info(f"Hashing prefix: {prefix} == {prefix_hash}")
    else:
        output_path = output_dir / f"emb__{mname}"
    output_path.mkdir(parents=True, exist_ok=True)

    writer = None

    # Load dataset in streaming mode
    ds = load_dataset(dataset_name, split=split, streaming=False)

    # Filter newspapers if provided
    if newspapers:
        selected_newspapers = [n.strip() for n in newspapers.split(",")]
        ds = ds.filter(lambda row: row["newspaper"] in selected_newspapers)
        logger.info(f"Subset: {ds.num_rows} articles from {selected_newspapers}")

    if max_articles:
        ds = ds.select(range(min(max_articles, ds.num_rows)))

    # Then call:
    ds = ds.map(
        preprocess,
        fn_kwargs={"model_name": model_name, "max_tokens": max_tokens, "prefix": prefix},
        num_proc=4
        )

    # Convert to Hugging Face Dataset and save
    ds.save_to_disk(output_dir / "preprocessed_8194")

    logger.info(f"✅ Saved preprocessed dataset with chunks to {output_dir/'preprocessed_8194'}")

if __name__ == "__main__":
    app()
