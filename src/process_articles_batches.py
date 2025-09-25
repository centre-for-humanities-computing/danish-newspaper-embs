import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
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


def chunk_sentences(sentences: list[str], max_tokens: int, model: SentenceTransformer) -> list[str]:
    output, current_chunk, chunk_len = [], [], 0
    for sentence in sentences:
        tokens = model.tokenize(sentence)
        seq_len = len(tokens["input_ids"])
        if chunk_len + seq_len > max_tokens:
            if len(current_chunk) == 0:
                output.extend(split_long_sentence(sentence, max_tokens, model))
            else:
                output.append(" ".join(current_chunk))
                current_chunk, chunk_len = [], 0
        current_chunk.append(sentence)
        chunk_len += seq_len
    if current_chunk:
        output.append(" ".join(current_chunk))
    return output


def split_long_sentence(sentence: str, max_tokens: int, model: SentenceTransformer) -> list[str]:
    words, parts, current_part, current_len = sentence.split(), [], [], 0
    for word in words:
        tokens = model.tokenize(word)
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


@app.command()
def main(
    dataset_name: str = typer.Option("JohanHeinsen/ENO", help="Hugging Face dataset name"),
    split: str = typer.Option("train", help="Which split to use"),
    output_dir: Path = typer.Option('../data/all/', help="Directory to save embeddings"),
    model_name: str = typer.Option("JohanHeinsen/Old_News_Segmentation_SBERT_V0.1", help="SentenceTransformer model"),
    prefix: str = typer.Option("Query: ", help="Prefix for each chunk"),
    prefix_description: str = typer.Option(None, help="Used in output dir name"),
    batch_articles: int = typer.Option(1024, help="Number of articles to preprocess before encoding"),
    batch_size: int = typer.Option(100, help="Batch size for each GPU process"),
    chunk_size: int = typer.Option(1000, help="Number of sentences sent to each worker at a time"),
    max_articles: int = typer.Option(5000, help="Limit total number of articles to process (for testing)"),
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

    # Arrow schema
    schema = pa.schema([
        ("article_id", pa.string()),
        ("date", pa.string()),
        ("nsp", pa.string()),
        ("chunk", pa.list_(pa.string())),
        ("embedding", pa.list_(pa.list_(pa.float32()))),
    ])

    parquet_file = output_path / "embeddings.parquet"
    writer = None

    # Load dataset in streaming mode
    ds = load_dataset(dataset_name, split=split, streaming=True)

    # Start multi-process pool (workers pinned to GPUs)
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2"])

    #for batch in tqdm(stream_batches(ds, batch_articles), desc="Processing batches"):

    processed_count = 0

    for batch in tqdm(stream_batches(ds, batch_articles), desc="Processing batches"):
        if max_articles and processed_count >= max_articles:
            break

        # If we're near the end, trim batch to avoid overshooting
        if max_articles:
            remaining = max_articles - processed_count
            if len(batch) > remaining:
                batch = batch[:remaining]    

        article_chunks, meta = [], []

        # Step 1: preprocess articles into chunks
        for row in batch:
            try:
                text_clean = clean_whitespace(row["text"])
                sentences = simple_sentencize(text_clean)
                chunks = chunk_sentences(sentences, max_tokens=max_tokens, model=model)
            except Exception as e:
                logger.error(f"Preprocessing error for article {row['id']}: {e}")
                continue

            chunk_inputs = [f"{prefix} {c}" if prefix else c for c in chunks]
            article_chunks.extend(chunk_inputs)
            meta.append((str(row["id"]), row.get("date", ""), row.get("newspaper", ""), chunks))

        # Step 2: encode all chunks via multi-process pool
        try:
            embs = model.encode(
                article_chunks,
                pool=pool,
                batch_size=batch_size,
                chunk_size=chunk_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            continue

        # Step 3: remap embeddings back to articles
        records, idx = [], 0
        for article_id, date, nsp, chunks in meta:
            n_chunks = len(chunks)
            chunk_embs = embs[idx: idx + n_chunks]
            idx += n_chunks
            records.append({
                "article_id": article_id,
                "date": date,
                "nsp": nsp,
                "chunk": [str(c) for c in chunks],
                "embedding": [list(map(float, emb)) for emb in chunk_embs],
            })

        table = pa.Table.from_pylist(records, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, schema=schema)
        writer.write_table(table)

        processed_count += len(batch)

    if writer is not None:
        writer.close()

    # Stop the pool
    model.stop_multi_process_pool(pool)

    logger.info(f"✅ Finished. Saved embeddings to {parquet_file}")


if __name__ == "__main__":
    app()
