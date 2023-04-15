"""

Neural IR with Sentence Transformers

@kylel

"""

import json
import logging
import os
from shutil import rmtree

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

import pyterrier as pt
import torch

if not pt.started():
    pt.init()

from ir_measures import Recall

# safe import
try:
    from hyde.basic_config import (
        BATCH_SIZE,
        HEAD_SIZE,
        NEU_MODEL_NAME,
        neu_index_path,
        neu_metricspath,
        neu_resultspath,
        raw_bookspath,
        raw_postspath,
    )
    from hyde.load_data.load_data_ours import load_books, load_posts_and_qrels
except ImportError:
    pass
from pyterrier_sentence_transformers import (
    SentenceTransformersIndexer,
    SentenceTransformersRetriever,
)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())


# load data
books = load_books(booksfile=raw_bookspath)
posts, qrels = load_posts_and_qrels(postsfile=raw_postspath)


# build neu index
NEU_indexer = SentenceTransformersIndexer(
    model_name_or_path=NEU_MODEL_NAME,
    index_path=neu_index_path,
    overwrite=True,
    normalize=False,
    text_attr=["text"],
)
NEU_indexer.index(books)


# define retriever
NEU_br = SentenceTransformersRetriever(
    model_name_or_path=NEU_MODEL_NAME,
    index_path=neu_index_path,
    device="cuda",
)

# test single query
search_results = NEU_br.search("Damien Graves")
search_results = search_results.join(books.set_index("docno"), on="docno")
logger.info(search_results)

# batch query
all_results = []
sub_posts = posts.head(HEAD_SIZE)
for batch_id, batch in tqdm(sub_posts.groupby(sub_posts.index // BATCH_SIZE)):
    batch_results = NEU_br.transform(batch)
    all_results.append(batch_results)
    logger.info(batch_results)
results = pd.concat(all_results)
results = results.join(books.set_index("docno"), on="docno")

# evaluate
metrics = pt.Utils.evaluate(
    results, qrels, metrics=[Recall @ 10, Recall @ 100, Recall @ 1000]
)

# save results
results.to_csv(neu_resultspath, index=False)

# save metrics
with open(neu_metricspath, "w") as f:
    json.dump(metrics, f, indent=4)

# clean up
rmtree(neu_index_path)
