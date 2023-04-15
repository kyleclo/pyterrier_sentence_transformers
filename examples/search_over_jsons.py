from pathlib import Path
import shutil
from typing import Optional
import platformdirs
import pyterrier as pt
import json

if not pt.started():
    pt.init()

from pyterrier_sentence_transformers import (
    SentenceTransformersRetriever,
    SentenceTransformersIndexer,
)
import ir_measures
from ir_measures import Recall

import pandas as pd
from tqdm import tqdm



import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()


DATASET_NAME = 'from_json'
NEU_MODEL_NAME = 'facebook/contriever-msmarco'
STA_MODEL_NAME = 'BM25'


class StripMarkup():
    # following https://github.com/terrier-org/pyterrier/issues/253
    def __init__(self):
        self.tokenizer = pt.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

    def __call__(self, text):
        return " ".join(self.tokenizer.getTokens(text))


def load_data(booksfile, postsfile):

    books = []
    with open(booksfile) as f_in:
        for line in f_in:
            book = json.loads(line)
            if 'tomt' in booksfile and 'tomt' in postsfile:
                books.append({
                    'docno': f"TOMT_{book['book_id']}",
                    # 'text': book['title']
                    'text': book['title'] + ' by ' + book['author'] if book['author'] else book['title']
                })
            elif ('ours' in booksfile and 'ours' in postsfile) or ('gold' in booksfile and 'gold' in postsfile):
                books.append({
                    'docno': f"OURS_{book['id']}",
                    'text': book['title'] + ' by ' + book['author'] if book['author'] else book['title']
                })
    # only for dense
    # books = pd.DataFrame.from_records(books)

    posts = []
    qrels = []
    with open(postsfile) as f_in:
        for line in f_in:
            post = json.loads(line)
            if 'tomt' in booksfile and 'tomt' in postsfile:
                posts.append({
                    'qid': f"TOMT_{post['id']}",
                    'query': post['title']['text']
                })
                qrels.append({
                    'qid': f"TOMT_{post['id']}",
                    'doc_id': f"TOMT_{post['gold_book']}",
                    'relevance': 1
                })
            elif ('ours' in booksfile and 'ours' in postsfile) or ('gold' in booksfile and 'gold' in postsfile):
                posts.append({
                    'qid': f"OURS_{post['title']['link']}",         # in ours, the ID of each post is the original URL
                    'query': post['comments'][0]['comment_text']    # in ours, the first comment is the query
                })
                qrels.append({
                    'qid': f"OURS_{post['title']['link']}",
                    'doc_id': f"OURS_{post['title']['book_id']}",
                    'relevance': 1
                })
    posts = pd.DataFrame.from_records(posts)
    qrels = pd.DataFrame.from_records(qrels)
    qrels['query_id'] = qrels['qid']            # need this for `ir_measures`

    markup_stripper = StripMarkup()
    posts = pt.apply.query(lambda r: markup_stripper(r.query))(posts)

    return books, posts, qrels


def load_retrievers_and_build_index(books: list):
    index_root = Path(
        platformdirs.user_cache_dir('pyterrier_sentence_transformers')
    ) / DATASET_NAME

    if index_root.exists():
        shutil.rmtree(index_root)

    # This is the neural indexer with sentence-transformers
    neu_index_path = index_root / NEU_MODEL_NAME.replace('/', '_')
    neu_index_path.mkdir(parents=True, exist_ok=True)
    indexer = SentenceTransformersIndexer(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path),
        overwrite=True,
        normalize=False,
        text_attr=['text']
    )
    indexer.index(books)

    # This is a classic statistical indexer
    sta_index_path = index_root / STA_MODEL_NAME
    sta_index_path.mkdir(parents=True, exist_ok=True)
    if not (sta_index_path / 'data.properties').exists():
        indexer = pt.IterDictIndexer(index_path=str(sta_index_path), blocks=True)
        indexref = indexer.index(books, fields=['text'])
        index = pt.IndexFactory.of(indexref)
    else:
        index = pt.IndexFactory.of(str(sta_index_path))

    # Retrievers (neural and statistical)
    neu_retr = SentenceTransformersRetriever(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path)
    )

    sta_retr = pt.BatchRetrieve(index, wmodel=STA_MODEL_NAME)

    return sta_retr, neu_retr


def predict_using_retriever(posts: pd.DataFrame, retriever, limit = None):

    runs = retriever.transform(posts)

    # runs = []
    # for _, post in posts.iterrows():
    #     preds = retriever.search(post['query'])
    #     preds = [pred for _, pred in preds.iterrows()]
    #     for pred in preds:
    #         run = {'query_id': post['qid'], 'doc_id': pred['docno'], 'score': pred['score']}
    #         runs.append(run)
    # runs = pd.DataFrame.from_records(runs)
    runs['query_id'] = runs['qid']              # required for ir_measures
    runs['doc_id'] = runs['docno']              # required for ir_measures

    if limit:
        return runs.loc[runs['rank'] < limit]          # e.g. limit=2 means returning rank0, rank1
    else:
        return runs


def spoof_posts_with_gpt3_titles(post_id_to_gpt3_10_gens: dict,
                                 num_gpt3_gens_to_use) -> pd.DataFrame:
    """
    post_id_to_gpt3_10_gens = {
        'TOMT_ies8mp': = [<str>, <str>, ..., <str>],
        ...
    }

    output:
        qid             |       query_0     |       query
    'TOMT_ies8mp__0'    |   <fake title>    |   <fake title lower() strip() rmpunct()>
    'TOMT_ies8mp__1'    |   <fake title>    |   <fake title lower() strip() rmpunct()>
    ...
    'TOMT_d7llps__3'    |   <fake title>    |   <fake title lower() strip() rmpunct()>
    """
    gpt3_spoofed_posts = []
    for post_id, gpt3_gens in post_id_to_gpt3_10_gens.items():
        for k, gpt3_gen in enumerate(gpt3_gens[:num_gpt3_gens_to_use]):
            gpt3_spoofed_posts.append({'qid': f"{post_id}__{k}", 'query': gpt3_gen})
    gpt3_spoofed_posts = pd.DataFrame.from_records(gpt3_spoofed_posts)
    markup_stripper = StripMarkup()
    gpt3_spoofed_posts = pt.apply.query(lambda r: markup_stripper(r.query))(gpt3_spoofed_posts)
    return gpt3_spoofed_posts


def spoof_qrels_with_gpt3_books(qrels: pd.DataFrame, gpt3_books: list) -> pd.DataFrame:
    for _, row in qrels.iterrows():
        {'qid': row['qid'], 'doc_id': row['doc_id'], 'relevance': 1}


def main():

    # load data
    books, posts, qrels = load_data(
        booksfile='/net/nfs2.s2-research/kylel/whatsthatbook/data/2022-05-30_14441_gold_books.jsonl',
        postsfile='/net/nfs2.s2-research/kylel/whatsthatbook/data/2022-05-30_14441_gold_posts.jsonl'
        # booksfile='/net/nfs2.s2-research/kylel/whatsthatbook/data/2022-05-30_tomt_books.jsonl',
        # postsfile='/net/nfs2.s2-research/kylel/whatsthatbook/data/2022-05-30_tomt_posts.jsonl'
    )
    print(f'Books: {len(books)}')
    print(f'Posts: {len(posts)}')

    # export data
    if False:
        books.to_csv('2022-05-30_14441_gold_books.csv', index=False)
        posts.to_csv('2022-05-30_14441_gold_posts.csv', index=False)
        qrels.to_csv('2022-05-30_14441_gold_qrels.csv', index=False)

    # load retrievers
    sta_retr, neu_retr = load_retrievers_and_build_index(books=books)

    # run the experiement
    exp = pt.Experiment(
        [sta_retr, neu_retr],
        posts,
        qrels,
        names=[STA_MODEL_NAME, NEU_MODEL_NAME],
        eval_metrics=[Recall@10, Recall@100, Recall@1000]
    )
    print(exp)


    # to search, use stat_retr.search(string) or neu_retr.search(string)
    mini_posts = posts.iloc[:500]
    sta_runs = predict_using_retriever(posts=mini_posts, retriever=sta_retr)
    neu_runs = predict_using_retriever(posts=mini_posts, retriever=neu_retr)


    # evaluate based on runs (make sure it's same as the Experimental framework)
    sta_eval = ir_measures.calc_aggregate([Recall @ 10, Recall @ 100, Recall @ 1000], qrels, sta_runs)
    neu_eval = ir_measures.calc_aggregate([Recall @ 10, Recall @ 100, Recall @ 1000], qrels, neu_runs)


    #
    # now to do the thing we're here for!
    #


    # we used GPT3 to generate book titles. what if we loaded them up & mapped them to books?
    # one way to do that is to format these titles like `queries` in `posts`
    # then run these through a retriever.
    post_id_to_gpt3_10_gens = {}
    with open(
        '/net/nfs/s2-research/kylel/whatsthatbook/data/2022-05-30_14441_chatgpt_preds_2023-03.jsonl'
        # '/net/nfs/s2-research/kylel/whatsthatbook/gpt_experiments/2022-05-30_tomt_queries_answered_by_gpt3_2023-01.jsonl'
    ) as f_in:
        for line in f_in:
            gen = json.loads(line)
            # post_id_to_gpt3_10_gens[f"TOMT_{gen['post_id']}"] = gen['contexts']
            post_id_to_gpt3_10_gens[f"OURS_{gen['post_id']}"] = gen['contexts']

    # this step spoofs a `posts` dataframe (i.e. sets queries = gpt3 titles)
    NUM_GPT3_GENS_TO_USE = 10
    NUM_RETRIEVED_PER_GEN = 3
    gpt3_spoofed_posts = spoof_posts_with_gpt3_titles(post_id_to_gpt3_10_gens=post_id_to_gpt3_10_gens,
                                                      num_gpt3_gens_to_use=NUM_GPT3_GENS_TO_USE)

    # this step runs these generated titles through retrievers to find corresp books
    sta_runs_gpt3 = predict_using_retriever(posts=gpt3_spoofed_posts, retriever=sta_retr, limit=NUM_RETRIEVED_PER_GEN)
    neu_runs_gpt3 = predict_using_retriever(posts=gpt3_spoofed_posts, retriever=neu_retr, limit=NUM_RETRIEVED_PER_GEN)

    # finally, let's reformat the data
    cleaned_sta_runs_gpt3 = pd.DataFrame()
    cleaned_sta_runs_gpt3['query_id'] = sta_runs_gpt3['query_id'].apply(lambda q: q.split('__')[0])
    cleaned_sta_runs_gpt3['query'] = sta_runs_gpt3['query']
    cleaned_sta_runs_gpt3['doc_id'] = sta_runs_gpt3['doc_id']
    cleaned_sta_runs_gpt3['score'] = sta_runs_gpt3['score']


    cleaned_neu_runs_gpt3 = pd.DataFrame()
    cleaned_neu_runs_gpt3['query_id'] = neu_runs_gpt3['query_id'].apply(lambda q: q.split('__')[0])
    cleaned_neu_runs_gpt3['doc_id'] = neu_runs_gpt3['doc_id']
    cleaned_neu_runs_gpt3['score'] = neu_runs_gpt3['score']

    # annoyingly, pyterrier drops trailing `_` in query IDs, so we need to fix that
    query_id_to_query = {post['qid']: post['query'] for _, post in posts.iterrows()}
    query = []
    for qids in cleaned_neu_runs_gpt3['query_id']:
        if qids not in query_id_to_query:
            qids = qids + '_'
            if qids not in query_id_to_query:
                raise Exception
        query.append(query_id_to_query[qids])
    cleaned_neu_runs_gpt3['query'] = query
    

    # evaluate!
    ir_measures.calc_aggregate([Recall @ 10], qrels, cleaned_sta_runs_gpt3)
    ir_measures.calc_aggregate([Recall @ 10], qrels, cleaned_neu_runs_gpt3)



    #
    #
    #
    #  OK now let's add the fake docs
    #
    #
    #


    gpt3_books = []
    with open(
        '/net/nfs/s2-research/kylel/whatsthatbook/data/2022-05-30_ours_books_chatgpt_2023-03.jsonl'
        # '/net/nfs/s2-research/kylel/whatsthatbook/data/2022-05-30_tomt_books_gpt3_2022-12.jsonl'
    ) as f_in:
        for line in tqdm(f_in):
            gpt3_book = json.loads(line)
            gpt3_books.append({'docno': f"OURS_{gpt3_book['book_id']}", 'text': gpt3_book['contexts'][0]})             # start experiments w only encoding 1 of the generations
            # gpt3_books.append({'docno': f"TOMT_{gpt3_book['book_id']}", 'text': gpt3_book['contexts'][0]})             # start experiments w only encoding 1 of the generations
    print(f'GPT3 Books: {len(gpt3_books)}')


    # reload retrievers
    sta_hyde_retr, neu_hyde_retr = load_retrievers_and_build_index(books=gpt3_books)

    exp_hyde = pt.Experiment(
        [sta_hyde_retr, neu_hyde_retr],
        posts,
        qrels,
        names=[STA_MODEL_NAME, NEU_MODEL_NAME],
        eval_metrics=[Recall@10, Recall@100, Recall@1000]
    )
    print(exp_hyde)

    sta_hyde_runs = predict_using_retriever(posts=posts, retriever=sta_hyde_retr)
    neu_hyde_runs = predict_using_retriever(posts=posts, retriever=neu_hyde_retr)

    sta_hyde_eval = ir_measures.calc_aggregate([Recall @ 10, Recall @ 100, Recall @ 1000], qrels, sta_hyde_runs)
    neu_hyde_eval = ir_measures.calc_aggregate([Recall @ 10, Recall @ 100, Recall @ 1000], qrels, neu_hyde_runs)



    #
    #
    #  OK, now to combine it all. hallucinated titles & hallucinated docs.
    #

    print(f'GPT3 generated titles per post: {len(post_id_to_gpt3_10_gens)}')

    # uses the spoofed posts from before
    assert gpt3_spoofed_posts

    # this step runs these generated titles through retrievers to find corresp books
    sta_runs_gpt3_hyde = predict_using_retriever(posts=gpt3_spoofed_posts, retriever=sta_hyde_retr, limit=NUM_RETRIEVED_PER_GEN)
    neu_runs_gpt3_hyde = predict_using_retriever(posts=gpt3_spoofed_posts, retriever=neu_hyde_retr, limit=NUM_RETRIEVED_PER_GEN)

    # finally, let's reformat the data
    cleaned_sta_hyde_runs_gpt3 = pd.DataFrame()
    cleaned_sta_hyde_runs_gpt3['query_id'] = sta_runs_gpt3_hyde['query_id'].apply(lambda q: q.split('__')[0])
    cleaned_sta_hyde_runs_gpt3['query'] = sta_runs_gpt3_hyde['query']
    cleaned_sta_hyde_runs_gpt3['doc_id'] = sta_runs_gpt3_hyde['doc_id']
    cleaned_sta_hyde_runs_gpt3['score'] = sta_runs_gpt3_hyde['score']


    cleaned_neu_hyde_runs_gpt3 = pd.DataFrame()
    cleaned_neu_hyde_runs_gpt3['query_id'] = neu_runs_gpt3_hyde['query_id'].apply(lambda q: q.split('__')[0])
    cleaned_neu_hyde_runs_gpt3['doc_id'] = neu_runs_gpt3_hyde['doc_id']
    cleaned_neu_hyde_runs_gpt3['score'] = neu_runs_gpt3_hyde['score']

    # annoyingly, pyterrier drops trailing `_` in query IDs, so we need to fix that
    query_id_to_query = {post['qid']: post['query'] for _, post in posts.iterrows()}
    query = []
    for qids in cleaned_neu_hyde_runs_gpt3['query_id']:
        if qids not in query_id_to_query:
            qids = qids + '_'
            if qids not in query_id_to_query:
                raise Exception
        query.append(query_id_to_query[qids])
    cleaned_neu_hyde_runs_gpt3['query'] = query
    


    # evaluate!
    ir_measures.calc_aggregate([Recall @ 10], qrels, cleaned_sta_hyde_runs_gpt3)
    ir_measures.calc_aggregate([Recall @ 10, Recall @ 100, Recall @ 1000], qrels, cleaned_neu_hyde_runs_gpt3)






if __name__ == '__main__':
    main()




# side script - test the ad hoc eval works
runs = []
for _, qrel in qrels.iterrows():
    run = {'query_id': qrel['query_id'], 'doc_id': qrel['doc_id'], 'score': qrel['relevance']}
    runs.append(run)
runs = pd.DataFrame.from_records(runs)
ir_measures.calc_aggregate([Recall@10, Recall@100, Recall@1000], qrels, runs)

