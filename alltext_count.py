#各分野の論文で1回以上出現
#TF-IDFが上位70%
#全論文でカウントし、出現頻度が上位2500語

import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# =========================
# tokenize
# =========================
def tokenize_line(line):
    return re.findall(r"\b\w+\b", line.lower())

# =========================
# main
# =========================
def main(input_folder, output_csv="vocab.csv", top_n=2500):

    files = [os.path.join(input_folder, f)
             for f in os.listdir(input_folder) if f.endswith(".txt")]

    total_docs = len(files)

    word_doc_count = Counter()
    word_freq_total = Counter()
    word_tfidf_per_doc = defaultdict(list)

    # =========================
    # 1st pass: freq + doc_freq
    # =========================
    for path in files:
        counter = Counter()

        with open(path, encoding="utf-8") as f:
            for line in f:
                counter.update(tokenize_line(line))

        for w, tf in counter.items():
            word_freq_total[w] += tf
            word_doc_count[w] += 1

    # =========================
    # IDF
    # =========================
    idf = {
        w: math.log((1 + total_docs) / (1 + df)) + 1
        for w, df in word_doc_count.items()
    }

    # =========================
    # TF-IDF per document
    # =========================
    for path in files:
        counter = Counter()

        with open(path, encoding="utf-8") as f:
            for line in f:
                counter.update(tokenize_line(line))

        for w, tf in counter.items():
            word_tfidf_per_doc[w].append(tf * idf[w])

    # =========================
    # DataFrame作成
    # =========================
    data = []

    for w in word_freq_total:

        tfidf_vals = word_tfidf_per_doc[w]

        data.append([
            w,
            word_freq_total[w],
            np.mean(tfidf_vals),
            np.var(tfidf_vals),
            word_doc_count[w]
        ])

    df = pd.DataFrame(
        data,
        columns=["word", "freq", "tfidf", "variance", "doc_freq"]
    )

    # =========================
    # フィルタ①：全48ファイルに出現
    # =========================
    df = df[df["doc_freq"] == total_docs]

    # =========================
    # フィルタ②：TF-IDF上位70%
    # =========================
    tfidf_threshold = df["tfidf"].quantile(0.3)
    df = df[df["tfidf"] >= tfidf_threshold]

    # =========================
    # ソート：freq降順
    # =========================
    df = df.sort_values(by="freq", ascending=False)

    # =========================
    # Top N
    # =========================
    df = df.head(top_n)

    # =========================
    # 出力
    # =========================
    df.to_csv(output_csv, index=False)

    # =========================
    # チェック
    # =========================
    print("=== TOP 20 ===")
    print(df.head(20))

    print("\n=== doc_freq check ===")
    print(df["doc_freq"].describe())


if __name__ == "__main__":
    main(input_folder=".")
