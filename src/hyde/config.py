"""

@kylel

"""
import os

DATASET_NAME = "from_json"
STA_MODEL_NAME = "BM25"
NEU_MODEL_NAME = "facebook/contriever-msmarco"

# data paths
booksfile = "2022-05-30_14441_gold_books.csv"
postsfile = "2022-05-30_14441_gold_posts.csv"
qrelsfile = "2022-05-30_14441_gold_qrels.csv"
resultsdir = "2022-05-30_14441_gold/"
os.makedirs(resultsdir, exist_ok=True)
index_dir = "./index"
os.makedirs(index_dir, exist_ok=True)

# sta model output paths
sta_resultsfile = f"{STA_MODEL_NAME}-results.csv"
sta_resultspath = os.path.join(resultsdir, sta_resultsfile)
sta_metricsfile = f"{STA_MODEL_NAME}-metrics.json"
sta_metricspath = os.path.join(resultsdir, sta_metricsfile)

# neu model output paths
neu_resultsfile = f"{NEU_MODEL_NAME}-results.csv"
neu_resultspath = os.path.join(resultsdir, neu_resultsfile)
neu_metricsfile = f"{NEU_MODEL_NAME}-metrics.json"
neu_metricspath = os.path.join(resultsdir, neu_metricsfile)

# indexer paths
sta_index_path = os.path.join(index_dir, STA_MODEL_NAME)
neu_index_path = os.path.join(index_dir, NEU_MODEL_NAME)

# inference parameters
BATCH_SIZE = 500
HEAD_SIZE = 1000
assert BATCH_SIZE <= HEAD_SIZE
