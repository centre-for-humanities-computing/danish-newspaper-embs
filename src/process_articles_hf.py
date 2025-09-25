import re
import hashlib
from pathlib import Path
import numpy as np
import json

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

import pyarrow as pa

app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def clean_whitespace(text: str) -> str:
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove excess spaces after punctuation
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # Strip leading and trailing spaces
    return text.strip()


def simple_sentencize(text: str) -> list:
    """
    Split text into sentences using punctuation as delimiter.
    """
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(sentences: list, max_tokens: int, model: SentenceTransformer) -> list:
    """
    Combine sentences into chunks so that the total token count per chunk is below `max_tokens`.
    """
    output = []
    current_chunk = []
    chunk_len = 0

    for sentence in sentences:
        tokens = model.tokenize(sentence)
        seq_len = len(tokens["input_ids"])

        if chunk_len + seq_len > max_tokens:
            # If the sentence alone is too long and current_chunk is empty,
            # split the sentence word-by-word.
            if len(current_chunk) == 0:
                parts = split_long_sentence(sentence, max_tokens=max_tokens, model=model)
                output.extend(parts)
            else:
                output.append(" ".join(current_chunk))
                current_chunk = []
                chunk_len = 0

        current_chunk.append(sentence)
        chunk_len += seq_len

    if current_chunk:
        output.append(" ".join(current_chunk))

    return output


def split_long_sentence(sentence: str, max_tokens: int, model: SentenceTransformer) -> list:
    """
    Split a long sentence into smaller parts on a word-by-word basis if its token length exceeds max_tokens.
    """
    words = sentence.split()
    parts = []
    current_part = []
    current_len = 0

    for word in words:
        tokens = model.tokenize(word)
        seq_len = len(tokens["input_ids"])

        if current_len + seq_len > max_tokens:
            parts.append(" ".join(current_part))
            current_part = []
            current_len = 0

        current_part.append(word)
        current_len += seq_len

    if current_part:
        parts.append(" ".join(current_part))

    return parts


# Function to find the maximum allowed tokens for the model
def find_max_tokens(tokenizer):
    """
    Determines the maximum token length for the tokenizer, ensuring it doesn't exceed a reasonable limit.
    """
    max_length = tokenizer.model_max_length
    if max_length > 9000:  # sometimes, they default to ridiculously high values, so we set a max
        max_length = 510
    # if we get an error, we set it to 512
    try:
        # Test the tokenizer with a dummy input
        tokenizer("This is a test sentence.")
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        max_length = 510  # fallback to a safe default
    return max_length


@app.command()
def main(
    dataset_name: str = typer.Option('JohanHeinsen/ENO', help="Hugging Face dataset name, e.g. 'JohanHeinsen/ENO'"),
    split: str = typer.Option("train", help="Which split of the dataset to use"),
    output_dir: Path = typer.Option(..., help="Directory where the processed dataset will be saved, should be in embeddings"),
    model_name: str = typer.Option('JohanHeinsen/Old_News_Segmentation_SBERT_V0.1', help="SentenceTransformer model name for inference"),
    prefix: str = typer.Option('Query: ', help="Optional prefix/instruction to add to each chunk before encoding"),
    prefix_description: str = typer.Option(None, help="Short description of the prefix (used in the output directory name)"),
):
    """
    This script loads a Hugging Face dataset, preprocesses and chunks texts,
    computes embeddings for each chunk, and saves the output dataset to disk.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True) # needed for jina

    # find max_tokens via tokenizer
    max_tokens = find_max_tokens(model.tokenizer) # will default to 510 if too high
    print(f"Max tokens for {model_name}: {max_tokens}")
    logger.info(f"Max tokens for {model_name}: {max_tokens}")

    # Build output path based on model name and optional prefix
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

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Load HF dataset
    ds = load_dataset(dataset_name, split=split)

    processed_articles = []

    for row in tqdm(ds, total=len(ds), desc="Processing articles"):
        article_id = row['id']
        text = row['text']
        nsp = row['newspaper']
        date = row['date']


        # Preprocessing: clean and split the text into sentences/chunks
        try:
            text_clean = clean_whitespace(text)
            sentences = simple_sentencize(text_clean)
            chunks = chunk_sentences(sentences, max_tokens=max_tokens, model=model)
        except Exception as e:
            logger.error(f"Preprocessing error for article_id {article_id}: {e}")
            continue

        # Inference: compute embeddings for each chunk
        try:
            embeddings = []
            for chunk in chunks:
                chunk_input = f"{prefix} {chunk}" if prefix else chunk
                emb = model.encode(chunk_input)
                embeddings.append(emb)
        except Exception as e:
            logger.error(f"Inference error for article_id {article_id}: {e}")
            continue

        # processed_articles.append({
        #     "article_id": article_id,
        #     "chunk": chunks,
        #     "embedding": [emb.tolist() for emb in embeddings]
        # })

        processed_articles.append({
            "article_id": str(article_id),
            "date": date,
            "nsp": nsp,
            "chunk": [str(chunk) for chunk in chunks],
            "embedding": [list(map(float, emb)) for emb in embeddings]
        })

    # # make sure they are the right format before dumping
    # sanitized_articles = []
    # for article in processed_articles:
    #     sanitized_article = {
    #         "article_id": str(article["article_id"]),
    #         "chunk": [str(chunk) for chunk in article["chunk"]],
    #         "embedding": [
    #             [float(x) for x in embedding] for embedding in article["embedding"]
    #         ]
    #     }
    #     sanitized_articles.append(sanitized_article)

#    # Export processed data as a Hugging Face dataset and save to disk
    dataset = Dataset.from_list(processed_articles)
    dataset.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")
    logger.info(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    app()
