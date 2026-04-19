#!/usr/bin/env python3

import os
import logging
import numpy as np
import pandas as pd
import itertools
import pickle
from gensim.models import Word2Vec, KeyedVectors

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename='run.log'
)

# =========================
# Utility
# =========================

def list_file(dp):
    return sorted([
        os.path.join(dp, f)
        for f in os.listdir(dp)
        if os.path.isfile(os.path.join(dp, f))
    ])

def save_pickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def cos_sim(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# =========================
# Linear mapping (closed form)
# =========================

def learn_linear_map(X, Z):
    """
    Solve W such that XW ≈ Z
    X: (n, d1)
    Z: (n, d2)
    """
    W, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return W  # (d1, d2)

# =========================
# Main
# =========================

def main():

    # -------------------------
    # Paths
    # -------------------------
    dp_src_txt = 'src_txt'
    dp_src_bin = 'src_bin'
    dp_pickle = 'pickle'
    dp_cossim = 'cossim'

    os.makedirs(dp_src_bin, exist_ok=True)
    os.makedirs(dp_pickle, exist_ok=True)
    os.makedirs(dp_cossim, exist_ok=True)

    wiki_path = 'vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt'

    # -------------------------
    # Load Wikipedia vectors
    # -------------------------
    logging.info("Loading Wikipedia model...")
    wiki_model = KeyedVectors.load_word2vec_format(wiki_path, binary=False)

    # =========================
    # 1. Train domain embeddings
    # =========================
    src_txt_list = list_file(dp_src_txt)

    for fp_txt in src_txt_list:
        fn = os.path.basename(fp_txt)
        fp_bin = os.path.join(dp_src_bin, fn + '.bin')

        if os.path.exists(fp_bin):
            continue

        logging.info(f"Training Word2Vec: {fn}")

        corpus = list(open(fp_txt, encoding='utf-8'))

        model = Word2Vec(
            sentences=[line.split() for line in corpus],
            vector_size=300,
            window=10,
            min_count=5,
            sg=1,
            negative=5,
            workers=4
        )

        model.wv.save_word2vec_format(fp_bin, binary=True)

    # =========================
    # 2. Vocabulary (must be expanded manually)
    # =========================
    vocab = ['analysis', 'model', 'design', 'method']  # ←ここは拡張前提

    # Wikipedia side vectors
    vocab = [v for v in vocab if v in wiki_model]

    X_wiki = np.array([wiki_model[w] for w in vocab])

    # =========================
    # 3. Learn mapping + cosine
    # =========================

    combis = list(itertools.combinations(src_txt_list, 2))
    results = []

    for i, (f1, f2) in enumerate(combis):

        logging.info(f"{i+1}/{len(combis)}: {f1} vs {f2}")

        model1 = KeyedVectors.load_word2vec_format(
            os.path.join(dp_src_bin, os.path.basename(f1) + '.bin'),
            binary=True
        )

        model2 = KeyedVectors.load_word2vec_format(
            os.path.join(dp_src_bin, os.path.basename(f2) + '.bin'),
            binary=True
        )

        vocab_valid = [w for w in vocab if w in model1 and w in model2]

        if len(vocab_valid) < 5:
            continue

        X1 = np.array([model1[w] for w in vocab_valid])
        X2 = np.array([model2[w] for w in vocab_valid])

        # mapping to Wikipedia space
        W1 = learn_linear_map(X1, X_wiki[:len(vocab_valid)])
        W2 = learn_linear_map(X2, X_wiki[:len(vocab_valid)])

        vec1 = np.array([model1[w] @ W1 for w in vocab_valid])
        vec2 = np.array([model2[w] @ W2 for w in vocab_valid])

        cos_list = [cos_sim(a, b) for a, b in zip(vec1, vec2)]

        results.append([os.path.basename(f1), os.path.basename(f2)] + cos_list)

    df = pd.DataFrame(results)
    df.to_csv('cossim.csv', index=False)

    print(df)


if __name__ == "__main__":
    main()
