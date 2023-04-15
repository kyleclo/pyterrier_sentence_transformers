"""

BM25 retrieval

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

if not pt.started():
    pt.init()

from ir_measures import Recall

# safe import
try:
    from hyde.basic_config import (
        BATCH_SIZE,
        HEAD_SIZE,
        STA_MODEL_NAME,
        raw_bookspath,
        raw_postspath,
        sta_index_path,
        sta_metricspath,
        sta_resultspath,
    )

    # from hyde.load_data.load_data_ours import load_books, load_posts_and_qrels
    from hyde.load_data.load_data_ours_fake_books import (
        load_books,
        load_posts_and_qrels,
    )
except ImportError:
    pass

# load data
books = load_books(booksfile=raw_bookspath)
posts, qrels = load_posts_and_qrels(postsfile=raw_postspath)

# build sta index
STA_indexer = pt.DFIndexer(
    sta_index_path,
    overwrite=True,
    normalize=False,
    text_attr=["text"],
)
STA_indexer.index(books["text"], books["docno"])

# define retriever
BM25_br = pt.BatchRetrieve(STA_indexer, wmodel=STA_MODEL_NAME)

# test single query
search_results = pt.BatchRetrieve(STA_indexer).search("Damien Graves")
search_results = search_results.join(books.set_index("docno"), on="docno")
logger.info(search_results)

# batch query
all_results = []
sub_posts = posts.head(HEAD_SIZE)
for batch_id, batch in tqdm(sub_posts.groupby(sub_posts.index // BATCH_SIZE)):
    batch_results = BM25_br.transform(batch)
    all_results.append(batch_results)
    logger.info(batch_results)
results = pd.concat(all_results)
results = results.join(books.set_index("docno"), on="docno")

# evaluate
metrics = pt.Utils.evaluate(
    results, qrels, metrics=[Recall @ 10, Recall @ 100, Recall @ 1000]
)

# save results
results.to_csv(sta_resultspath, index=False)

# save metrics
with open(sta_metricspath, "w") as f:
    json.dump(metrics, f, indent=4)

# index cleanup
rmtree(sta_index_path)
