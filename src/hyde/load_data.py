"""

Load data

@kylel

"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)


# read books
def load_books_from_csv(booksfile):
    _books = pd.read_csv(booksfile)
    logger.info(f"Number of books: {len(_books)}")
    books = _books.dropna(subset=["text"])
    if books.shape != _books.shape:
        logger.warning(
            f"Dropped {len(_books) - len(books)} books with missing text"
        )
    return books


# read posts
def load_posts_from_csv(postsfile):
    _posts = pd.read_csv(postsfile)
    logger.info(f"Number of posts: {len(_posts)}")
    posts = _posts.dropna(subset=["query"])
    if posts.shape != _posts.shape:
        logger.warning(
            f"Dropped {len(_posts) - len(posts)} posts with missing text"
        )
    return posts


# read qrels
def load_qrels_from_csv(qrelsfile):
    _qrels = pd.read_csv(qrelsfile)
    logger.info(f"Number of qrels: {len(_qrels)}")
    qrels = _qrels.dropna(subset=["qid", "query_id"])
    if qrels.shape != _qrels.shape:
        logger.warning(
            f"Dropped {len(_qrels) - len(qrels)} qrels with missing text"
        )
    return qrels
