#各分野の論文で5回以上出現
#TF-IDFが上位50%
#全論文でカウントし、出現頻度が上位2000語

import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# =========================
# Generator tokenizer（超重要）
# =========================
def tokenize_stream(text):
    for match in re.finditer(r"\b\w+\b", text.lower()):
        yield match.group(0)

# =========================
# Main
# =========================
def main(input_folder=r"C:/Users/kanappe/Desktop/txt", output_csv="vocab.csv", top_n=2000):

    files = [os.path.join(input_folder, f)
             for f in os.listdir(input_folder) if f.endswith(".txt")]

    total_docs = 0
    word_doc_count = Counter()
    word_freq_total = Counter()

    # -------------------------
    # 1st pass（逐次）
    # -------------------------
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        total_docs += 1

        counter = Counter(tokenize_stream(text))

        for w in counter:
            word_doc_count[w] += 1
            word_freq_total[w] += counter[w]

    # -------------------------
    # IDF
    # -------------------------
    idf = {
        w: math.log((1 + total_docs) / (1 + df)) + 1
        for w, df in word_doc_count.items()
    }

    # -------------------------
    # 2nd pass（逐次）
    # -------------------------
    word_tfidf_scores = defaultdict(list)

    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        counter = Counter(tokenize_stream(text))

        for w, tf in counter.items():
            word_tfidf_scores[w].append(tf * idf[w])

    # -------------------------
    # DataFrame
    # -------------------------
    data = []

    for w in word_freq_total:
        tfidf_vals = word_tfidf_scores[w]

        data.append([
            w,
            word_doc_count[w],
            word_freq_total[w],
            np.mean(tfidf_vals),
            np.var(tfidf_vals)
        ])

    df = pd.DataFrame(
        data,
        columns=["word", "doc_freq", "freq", "tfidf", "variance"]
    )

    # -------------------------
    # Filter
    # -------------------------
    df = df[df["doc_freq"] >= 5]

    threshold = df["tfidf"].median()
    df = df[df["tfidf"] >= threshold]

    df = df.sort_values(by="freq", ascending=False).head(top_n)

    df.to_csv(output_csv, index=False)

    print(df.head(20))


if __name__ == "__main__":
    main()
