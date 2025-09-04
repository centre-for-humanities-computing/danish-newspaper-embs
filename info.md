# setup
- run from the command line:
pip install typer[all] loguru tqdm pandas datasets sentence-transformers einops
cd danish-newspaper-embs

# Embeddings

- To get embeddings, run: 
python src/process_articles.py --input-csv data/sample_1500.csv --output-dir data/test_task/raw --model <insertname>

the models used in our runs were: 
- intfloat/multilingual-e5-large
- MiMe-MeMo/MeMo-BERT-03
- jinaai/jina-embeddings-v3

Script runs one at a time and saves the (raw) output in data/raw

# Pooling (to get the final embeddings)

run (specifying output dir and input embeddings)(each model tested below)
- python src/mean_pooling.py data/test_task/raw/emb__intfloat__multilingual-e5-large_597369d1 data/pooled/e-5/
- python src/mean_pooling.py data/test_task/raw/emb__MiMe-MeMo__MeMo-BERT-03_597369d1 data/pooled/memo/
- python src/mean_pooling.py data/test_task/raw/emb__jinaai__jina-embeddings-v3_597369d1 data/pooled/jina/
- python src/mean_pooling.py data/test_task/raw/emb__OrdalieTech__Solon-embeddings-large-0.1_597369d1 data/pooled/solon/

Script runs one at a time and saves the (pooled) output in data/pooled/<nickname>