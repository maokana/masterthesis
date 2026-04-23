#各分野の論文で1回以上出現
#全論文でカウントし、出現頻度が上位2500語

import os
import re
import numpy as np
import pandas as pd
from collections import Counter

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def main(input_folder=".", output_csv="vocab.csv"):

    files = [os.path.join(input_folder, f)
             for f in os.listdir(input_folder)
             if f.endswith(".txt")]

    total_docs = len(files)

    doc_counters = []
    word_doc_freq = Counter()   # ← これを追加

    # =========================
    # 1st pass
    # =========================
    for path in files:
        counter = Counter()

        with open(path, encoding="utf-8") as f:
            for line in f:
                counter.update(tokenize(line))

        doc_counters.append(counter)

        # ★ 前のコード方式（重要）
        for w in counter:
            word_doc_freq[w] += 1

    # 全単語集合
    all_words = set(word_doc_freq.keys())

    data = []

    # =========================
    # 単語ごとに集計
    # =========================
    for w in all_words:

        doc_freq = word_doc_freq[w]

        # ★ 前と同じ条件
        if doc_freq == total_docs:

            tf_list = []

            for counter in doc_counters:
                tf_list.append(counter.get(w, 0))

            freq = sum(tf_list)
            mean_tf = np.mean(tf_list)
            variance = np.var(tf_list)

            # CV
            if mean_tf > 0:
                cv = np.std(tf_list) / mean_tf
            else:
                cv = 0

            data.append([
                w,
                freq,
                mean_tf,
                variance,
                cv,
                doc_freq
            ])

    df = pd.DataFrame(
        data,
        columns=["word", "freq", "mean_tf", "variance", "cv", "doc_freq"]
    )

    df = df.sort_values(by="freq", ascending=False)

    df.to_csv(output_csv, index=False)

    print("total_docs:", total_docs)
    print("result size:", len(df))
    print(df.head(20))


if __name__ == "__main__":
    main(".")

