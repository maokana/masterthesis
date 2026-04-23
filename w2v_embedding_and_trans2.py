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
        if f.endswith(".txt") and os.path.isfile(os.path.join(dp, f))
    ])

def cos_sim(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# =========================
# Linear mapping
# =========================

def learn_linear_map(X, Z):
    W, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return W

# =========================
# Main
# =========================

def main():

    dp_src_txt = 'src_txt'
    dp_src_bin = 'src_bin'
    wiki_path = 'vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt'

    os.makedirs(dp_src_bin, exist_ok=True)

    # -------------------------
    # Wikipedia読み込み
    # -------------------------
    logging.info("Loading Wikipedia model...")
    wiki_model = KeyedVectors.load_word2vec_format(wiki_path, binary=False)

    # =========================
    # 1. Word2Vec学習
    # =========================
    src_txt_list = list_file(dp_src_txt)

    for fp_txt in src_txt_list:
        fn = os.path.basename(fp_txt)
        fp_bin = os.path.join(dp_src_bin, fn + '.bin')

        if os.path.exists(fp_bin):
            continue

        logging.info(f"Training: {fn}")

        with open(fp_txt, encoding='utf-8') as f:
            corpus = [line.split() for line in f]

        model = Word2Vec(
            sentences=corpus,
            vector_size=300,
            window=10,
            min_count=5,
            sg=1,
            negative=5,
            workers=4
        )

        model.wv.save_word2vec_format(fp_bin, binary=True)

    # =========================
    # 2. アンカー語（CV低い語を推奨）
    # =========================
    df_vocab = pd.read_csv("vocab.csv")

    # CVが低い＝安定語をアンカーに
    anchor_vocab = df_vocab.sort_values("cv").head(200)["word"].tolist()

    # Wikipediaに存在する語だけ残す
    anchor_vocab = [w for w in anchor_vocab if w in wiki_model]

    X_wiki = np.array([wiki_model[w] for w in anchor_vocab])

    # =========================
    # 3. analyse_list（比較対象）
    # =========================
    analyse_list = [
        'analysis','model','design','figure','method',
        'parameter','system','result','effect','data'
    ]

    # =========================
    # 4. Cos類似度計算
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

        # -------------------------
        # アンカー語の共通部分
        # -------------------------
        anchor_valid = [
            w for w in anchor_vocab
            if w in model1 and w in model2
        ]

        if len(anchor_valid) < 50:
            logging.warning("anchor too small, skip")
            continue

        X1 = np.array([model1[w] for w in anchor_valid])
        X2 = np.array([model2[w] for w in anchor_valid])
        Z = np.array([wiki_model[w] for w in anchor_valid])

        # -------------------------
        # 線形変換
        # -------------------------
        W1 = learn_linear_map(X1, Z)
        W2 = learn_linear_map(X2, Z)

        # -------------------------
        # analyse_listで比較
        # -------------------------
        analyse_valid = [
            w for w in analyse_list
            if w in model1 and w in model2 and w in wiki_model
        ]

        if len(analyse_valid) == 0:
            continue

        vec1 = np.array([model1[w] @ W1 for w in analyse_valid])
        vec2 = np.array([model2[w] @ W2 for w in analyse_valid])

        cos_list = [cos_sim(a, b) for a, b in zip(vec1, vec2)]

        results.append(
            [os.path.basename(f1), os.path.basename(f2)] + cos_list
        )

    # =========================
    # 出力
    # =========================
    columns = ["file1", "file2"] + analyse_valid
    df = pd.DataFrame(results, columns=columns)

    df.to_csv("cossim.csv", index=False)

    print(df.head())


if __name__ == "__main__":
    main()
