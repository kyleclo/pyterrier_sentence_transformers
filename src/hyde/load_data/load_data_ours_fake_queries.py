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
    from hyde.configs.fake_posts_config import (
        NUM_GPT3_GENS_TO_USE,
        NUM_RETRIEVED_PER_GEN,
    )
    from hyde.load_data.load_data_ours import load_books, load_posts_and_qrels
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


def load_fake_posts(postsfile):
    # load fake posts
    post_id_to_fake_posts = {}
    with open(postsfile) as f_in:
        for line in f_in:
            post = json.loads(line)
            post_id_to_fake_posts[f"OURS_{post['post_id']}"] = post["contexts"]
    # convert to posts format
    posts = []
    for post_id, fake_posts in post_id_to_fake_posts.items():
        for k, fake_post in enumerate(fake_posts[:NUM_GPT3_GENS_TO_USE]):
            posts.append({"qid": f"{post_id}__{k}", "query": fake_post})
    # usual processing
    _posts_df = pd.DataFrame.from_records(posts)
    logger.info(f"Number of posts: {len(_posts_df)}")
    markup_stripper = StripMarkup()
    _posts_df = pt.apply.query(lambda r: markup_stripper(r.query))(_posts_df)
    posts_df = _posts_df.dropna(subset=["query"])
    if posts_df.shape != _posts_df.shape:
        logger.warning(
            f"Dropped {len(_posts_df) - len(posts_df)} posts with missing text"
        )
    return posts_df
