"""

@kylel

"""
import os

DATASET_NAME = "2022-05-30_14441_gold"
TASK_SETUP = "standard"
STA_MODEL_NAME = "BM25"
NEU_MODEL_NAME = "facebook/contriever-msmarco"

# raw data paths
data_dir = "/net/nfs2.s2-research/kylel/whatsthatbook/data/"
raw_bookspath = os.path.join(data_dir, "2022-05-30_14441_gold_books.jsonl")
raw_postspath = os.path.join(data_dir, "2022-05-30_14441_gold_posts.jsonl")

resultsdir = "/net/nfs2.s2-research/kylel/whatsthatbook/results/"
os.makedirs(resultsdir, exist_ok=True)

# index paths
index_dir = "./index"
os.makedirs(index_dir, exist_ok=True)

# sta model output paths
sta_resultsfile = f"{DATASET_NAME}-{TASK_SETUP}-{STA_MODEL_NAME}-results.csv"
sta_resultspath = os.path.join(resultsdir, sta_resultsfile)
sta_metricsfile = f"{DATASET_NAME}-{TASK_SETUP}-{STA_MODEL_NAME}-metrics.json"
sta_metricspath = os.path.join(resultsdir, sta_metricsfile)

# neu model output paths
neu_resultsfile = f"{DATASET_NAME}-{TASK_SETUP}-{NEU_MODEL_NAME.replace('/', '_')}-results.csv"
neu_resultspath = os.path.join(resultsdir, neu_resultsfile)
neu_metricsfile = f"{DATASET_NAME}-{TASK_SETUP}-{NEU_MODEL_NAME.replace('/', '_')}-metrics.json"
neu_metricspath = os.path.join(resultsdir, neu_metricsfile)

# indexer paths
sta_index_path = os.path.join(index_dir, STA_MODEL_NAME)
neu_index_path = os.path.join(index_dir, NEU_MODEL_NAME)

# inference parameters
BATCH_SIZE = 500
HEAD_SIZE = 100000  # all
assert BATCH_SIZE <= HEAD_SIZE
