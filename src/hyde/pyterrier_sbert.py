"""

Neural IR with Sentence Transformers

@kylel

"""

import json
import logging
import os

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

from hyde.config import (
    BATCH_SIZE,
    HEAD_SIZE,
    NEU_MODEL_NAME,
    booksfile,
    neu_index_path,
    neu_metricspath,
    neu_resultspath,
    postsfile,
    qrelsfile,
)
from hyde.load_data import (
    load_books_from_csv,
    load_posts_from_csv,
    load_qrels_from_csv,
)
from pyterrier_sentence_transformers import (
    SentenceTransformersIndexer,
    SentenceTransformersRetriever,
)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())


# load data
books = load_books_from_csv(booksfile)
posts = load_posts_from_csv(postsfile)
qrels = load_qrels_from_csv(qrelsfile)


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
