import re
import hashlib
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import Dataset, load_dataset
from text_embeddings_inference import Client


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


def chunk_sentences(sentences: list[str], max_tokens: int) -> list[str]:
    """
    Naive chunking: just split sentences into groups until token budget is hit.
    TEI doesn’t expose tokenizer length, so keep max_tokens fixed (default 510).
    """
    output, current_chunk, chunk_len = [], [], 0
    for sentence in sentences:
        # estimate length as words (rough proxy for tokens)
        seq_len = len(sentence.split())
        if chunk_len + seq_len > max_tokens:
            if current_chunk:
                output.append(" ".join(current_chunk))
                current_chunk, chunk_len = [], 0
        current_chunk.append(sentence)
        chunk_len += seq_len
    if current_chunk:
        output.append(" ".join(current_chunk))
    return output


@app.command()
def main(
    dataset_name: str = typer.Option("JohanHeinsen/ENO", help="Hugging Face dataset name"),
    split: str = typer.Option("train", help="Which split to use"),
    output_dir: Path = typer.Option(..., help="Where to save processed dataset"),
    tei_url: str = typer.Option("http://localhost:8080", help="URL of TEI server"),
    newspapers: str = typer.Option(None, help="Comma-separated list of newspaper names to include"),
    prefix: str = typer.Option("Query: ", help="Prefix for each chunk"),
    prefix_description: str = typer.Option(None, help="Used in output directory name"),
    batch_size: int = typer.Option(64, help="Batch size for TEI client"),
    max_tokens: int = typer.Option(510, help="Max tokens per chunk (approximate)"),
    max_articles: int = typer.Option(None, help="Limit for testing"),
    save_every: int = typer.Option(10000, help="Save every N articles"),
):
    """
    Preprocess Hugging Face dataset into chunks, call TEI server for embeddings,
    and save results incrementally as Hugging Face datasets.
    """
    client = Client(tei_url)

    # Build output path
    if prefix:
        if prefix_description:
            output_path = output_dir / f"emb__tei_{prefix_description}"
        else:
            prefix_hash = hash_prompt(prefix)
            output_path = output_dir / f"emb__tei_{prefix_hash}"
            logger.info(f"Hashing prefix: {prefix} == {prefix_hash}")
    else:
        output_path = output_dir / "emb__tei"
    output_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)

    # Filter newspapers if provided
    if newspapers:
        selected_newspapers = [n.strip() for n in newspapers.split(",")]
        ds = ds.filter(lambda row: row["newspaper"] in selected_newspapers)
        logger.info(f"Subset: {len(ds)} articles from {selected_newspapers}")

    processed_articles = []
    file_index = 0

    for idx, row in enumerate(tqdm(ds, total=len(ds), desc="Processing articles")):
        if max_articles and idx >= max_articles:
            break

        article_id, text, nsp, date = row["id"], row["text"], row["newspaper"], row["date"]

        # Preprocess
        try:
            text_clean = clean_whitespace(text)
            sentences = simple_sentencize(text_clean)
            chunks = chunk_sentences(sentences, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Preprocessing error for article_id {article_id}: {e}")
            continue

        if not chunks:
            continue

        # Build inputs with prefix
        chunk_inputs = [f"{prefix} {c}" if prefix else c for c in chunks]

        # Call TEI server
        try:
            embeddings = client.encode(chunk_inputs, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Inference error for article_id {article_id}: {e}")
            continue

        processed_articles.append({
            "article_id": str(article_id),
            "date": date,
            "nsp": nsp,
            "chunk": [str(c) for c in chunks],
            "embedding": embeddings,
        })

        # Save periodically
        if (idx + 1) % save_every == 0:
            ds_chunk = Dataset.from_list(processed_articles)
            chunk_path = output_path / f"part-{file_index:05d}"
            ds_chunk.save_to_disk(chunk_path)
            logger.info(f"💾 Saved {len(processed_articles)} articles to {chunk_path}")
            processed_articles = []
            file_index += 1

    # Save leftovers
    if processed_articles:
        ds_chunk = Dataset.from_list(processed_articles)
        chunk_path = output_path / f"part-{file_index:05d}"
        ds_chunk.save_to_disk(chunk_path)
        logger.info(f"💾 Saved {len(processed_articles)} articles to {chunk_path}")

    logger.info(f"✅ Finished. All parts saved in {output_path}")


if __name__ == "__main__":
    app()
