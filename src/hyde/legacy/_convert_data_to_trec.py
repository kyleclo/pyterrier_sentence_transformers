"""

Convert Whatsthatbook data to TREC format

@kylel

"""

import json
import os

import pandas as pd
import tokenizers

# safe import
try:
    from hyde.basic_config import (
        raw_bookspath,
        raw_postspath,
        trec_bookspath,
        trec_postspath,
        trec_qrelspath,
    )
except ImportError:
    pass


def convert_tomt_books_posts_to_trec(booksfile, postsfile, outdir):
    pass


def convert_ours_books_posts_to_trec(
    raw_bookspath,
    raw_postspath,
    trec_bookspath,
    trec_postspath,
    trec_qrelspath,
):
    books = []
    with open(raw_bookspath) as f_in:
        for line in f_in:
            book = json.loads(line)
            books.append(
                {
                    "docno": f"OURS_{book['id']}",
                    "text": book["title"] + " by " + book["author"]
                    if book["author"]
                    else book["title"],
                }
            )

    posts = []
    qrels = []
    with open(raw_postspath) as f_in:
        for line in f_in:
            post = json.loads(line)
            # in ours, the ID of each post is the original URL
            # in ours, the first comment is the query
            posts.append(
                {
                    "qid": f"OURS_{post['title']['link']}",
                    "query": post["comments"][0]["comment_text"],
                }
            )
            qrels.append(
                {
                    "qid": f"OURS_{post['title']['link']}",
                    "doc_id": f"OURS_{post['title']['book_id']}",
                    "relevance": 1,
                }
            )
    posts = pd.DataFrame.from_records(posts)
    qrels = pd.DataFrame.from_records(qrels)
    qrels["query_id"] = qrels["qid"]  # need this for `ir_measures`

    sm = StripMarkup()
    posts["query"] = sm.bulk(posts["query"].values.tolist())

    books_df = pd.DataFrame.from_records(books)
    posts_df = pd.DataFrame.from_records(posts)
    qrels_df = pd.DataFrame.from_records(qrels)

    # save to TREC format
    books_df.to_csv(trec_bookspath, sep="\t", index=False)
    posts_df.to_csv(trec_postspath, sep="\t", index=False)
    qrels_df.to_csv(trec_qrelspath, sep="\t", index=False)


def convert_ours_fake_books_posts_to_trec(booksfile, postsfile, outdir):
    pass


def convert_ours_books_fake_posts_to_trec(booksfile, postsfile, outdir):
    pass


def convert_ours_fake_books_fake_posts_to_trec(booksfile, postsfile, outdir):
    pass


convert_ours_books_posts_to_trec(
    raw_bookspath=raw_bookspath,
    raw_postspath=raw_postspath,
    trec_bookspath=trec_bookspath,
    trec_postspath=trec_postspath,
    trec_qrelspath=trec_qrelspath,
)
