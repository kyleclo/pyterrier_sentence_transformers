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

try:
    from hyde.load_data.load_data_ours import load_posts_and_qrels
except ImportError:
    pass


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
