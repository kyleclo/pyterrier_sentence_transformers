"""

@kylel

"""


import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

import json

import pyterrier as pt

if not pt.started():
    pt.init()


class StripMarkup:
    # following https://github.com/terrier-org/pyterrier/issues/253
    def __init__(self):
        self.tokenizer = pt.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

    def __call__(self, text):
        return " ".join(self.tokenizer.getTokens(text))


def load_books(booksfile):
    books = []
    with open(booksfile) as f_in:
        for line in f_in:
            book = json.loads(line)
            books.append(
                {
                    "docno": f"OURS_{book['book_id']}",
                    "text": book["contexts"][
                        0
                    ],  # only experimenting w the first (0th) generation for now
                }
            )
    _books_df = pd.DataFrame.from_records(books)
    logger.info(f"Number of books: {len(_books_df)}")
    books_df = _books_df.dropna(subset=["text"])
    if books_df.shape != _books_df.shape:
        logger.warning(
            f"Dropped {len(_books_df) - len(books_df)} books with missing text"
        )
    return books_df


def load_posts_and_qrels(postsfile):
    posts = []
    qrels = []
    with open(postsfile) as f_in:
        for line in f_in:
            post = json.loads(line)
            posts.append(
                {
                    "qid": f"OURS_{post['title']['link']}",  # in ours, the ID of each post is the original URL
                    "query": post["comments"][0][
                        "comment_text"
                    ],  # in ours, the first comment is the query
                }
            )
            qrels.append(
                {
                    "qid": f"OURS_{post['title']['link']}",
                    "doc_id": f"OURS_{post['title']['book_id']}",
                    "relevance": 1,
                }
            )
    _posts_df = pd.DataFrame.from_records(posts)
    logger.info(f"Number of posts: {len(_posts_df)}")
    markup_stripper = StripMarkup()
    _posts_df = pt.apply.query(lambda r: markup_stripper(r.query))(_posts_df)
    posts_df = _posts_df.dropna(subset=["query"])
    if posts_df.shape != _posts_df.shape:
        logger.warning(
            f"Dropped {len(_posts_df) - len(posts_df)} posts with missing text"
        )

    _qrels_df = pd.DataFrame.from_records(qrels)
    _qrels_df["query_id"] = _qrels_df["qid"]  # need this for `ir_measures`
    logger.info(f"Number of qrels: {len(_qrels_df)}")
    qrels_df = _qrels_df.dropna(subset=["qid", "query_id"])
    if qrels_df.shape != _qrels_df.shape:
        logger.warning(
            f"Dropped {len(_qrels_df) - len(qrels_df)} qrels with missing text"
        )
    return posts_df, qrels_df
