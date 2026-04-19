#各分野の論文で5回以上出現
#TF-IDFが上位50%
#全論文でカウントし、出現頻度が上位2000語

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict

# =========================
# Load corpus
# =========================

def load_corpus(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    docs = []
    labels = []

    for f in files:
        path = os.path.join(folder, f)
        with open(path, encoding="utf-8") as file:
            docs.append(file.read())
            labels.append(f.replace(".txt", ""))

    return docs, labels


# =========================
# Main
# =========================

def main(input_folder="corpus", output_csv="vocab.csv", top_n=2000):

    # -------------------------
    # Load data
    # -------------------------
    docs, labels = load_corpus(input_folder)

    # -------------------------
    # TF-IDF
    # -------------------------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(docs)
    terms = np.array(vectorizer.get_feature_names_out())

    # -------------------------
    # Word statistics
    # -------------------------
    word_doc_count = Counter()
    word_freq_total = Counter()
    word_tfidf_scores = defaultdict(list)

    # per-document processing
    for i, doc in enumerate(docs):

        row = tfidf_matrix[i].toarray().flatten()

        words_in_doc = doc.lower().split()
        counter = Counter(words_in_doc)

        for w in counter:
            word_doc_count[w] += 1
            word_freq_total[w] += counter[w]

        for j, w in enumerate(terms):
            if row[j] > 0:
                word_tfidf_scores[w].append(row[j])

    # -------------------------
    # Build table
    # -------------------------
    data = []

    for w in terms:

        doc_count = word_doc_count[w]
        freq = word_freq_total[w]

        tfidf_vals = word_tfidf_scores.get(w, [])

        if len(tfidf_vals) == 0:
            continue

        tfidf_mean = np.mean(tfidf_vals)
        tfidf_var = np.var(tfidf_vals)

        data.append([
            w,
            doc_count,
            freq,
            tfidf_mean,
            tfidf_var
        ])

    df = pd.DataFrame(
        data,
        columns=["word", "doc_freq", "freq", "tfidf", "variance"]
    )

    # -------------------------
    # Filter conditions
    # -------------------------

    # (1) all domains >= 5 occurrences
    df = df[df["doc_freq"] >= 5]

    # (2) top 50% TF-IDF
    threshold = df["tfidf"].median()
    df = df[df["tfidf"] >= threshold]

    # -------------------------
    # Sort by frequency
    # -------------------------
    df = df.sort_values(by="freq", ascending=False)

    # -------------------------
    # Top N
    # -------------------------
    df = df.head(top_n)

    # -------------------------
    # Save
    # -------------------------
    df.to_csv(output_csv, index=False)

    print(df.head(20))


if __name__ == "__main__":
    main()
