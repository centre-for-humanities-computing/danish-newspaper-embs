# setup
- run from the command line:
pip install typer[all] loguru tqdm pandas datasets sentence-transformers einops
cd danish-newspaper-embs

# Embeddings

- To get embeddings, run: 
python src/process_articles.py --input-csv data/test_task/sample_500.csv --output-dir data/test_task/raw --model-name <insertname>

the models used in our runs were: 
- intfloat/multilingual-e5-large
- MiMe-MeMo/MeMo-BERT-03
- jinaai/jina-embeddings-v3
- JohanHeinsen/Old_News_Segmentation_SBERT_V0.1
- BAAI/bge-m3
- google/embeddinggemma-300m

Script runs one at a time and saves the (raw) output in data/raw

# Pooling (to get the final embeddings)

run (specifying output dir and input embeddings)(each model tested below)
- python src/mean_pooling.py data/test_task/raw/emb__intfloat__multilingual-e5-large_597369d1 data/test_task/pooled/e-5/
- python src/mean_pooling.py data/test_task/raw/emb__MiMe-MeMo__MeMo-BERT-03_597369d1 data/test_task/pooled/memo/
- python src/mean_pooling.py data/test_task/raw/emb__jinaai__jina-embeddings-v3_597369d1 data/test_task/pooled/jina/
- python src/mean_pooling.py data/test_task/raw/emb__JohanHeinsen__Old_News_Segmentation_SBERT_V0.1_597369d1/ data/test_task/pooled/old/
- python src/mean_pooling.py data/test_task/raw/emb__BAAI__bge-m3_597369d1/ data/test_task/pooled/bge/
- python src/mean_pooling.py data/test_task/raw/emb__google__embeddinggemma-300m_597369d1/ data/test_task/pooled/gemma/

Script runs one at a time and saves the (pooled) output in data/pooled/<nickname>