# =========================
# 最初に実行するコマンド
#python3 -m venv venv
#source venv/bin/activate
#pip install --upgrade pip
#pip install numpy==1.24.4
#pip install spacy==3.5.4 scikit-learn pandas
#python -m spacy download en_core_web_sm
# =========================



import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# モデル
# =========================

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 設定
# =========================

INPUT_DIR = "input"
BATCH_SIZE = 5000  # spaCy分割サイズ

# =========================
# 名詞抽出（分割処理）
# =========================

def extract_nouns_large_text(text):
    nouns = []

    # 長文を分割
    chunks = [text[i:i+BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]

    for chunk in chunks:
        doc = nlp(chunk.lower())

        for token in doc:
            if token.pos_ == "NOUN" and len(token.text) > 2:
                nouns.append(token.lemma_)

    return nouns

# =========================
# データ読み込み
# =========================

journal_nouns = {}
journal_names = []

files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

for file in files:
    name = os.path.basename(file).replace(".txt", "")

    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    nouns = extract_nouns_large_text(text)

    journal_nouns[name] = nouns
    journal_names.append(name)

# =========================
# 単語ごとの出現ジャーナル記録
# =========================

word_to_journals = defaultdict(set)
word_to_embeddings = defaultdict(list)

# =========================
# embedding作成（ジャーナル単位）
# =========================

journal_embeddings = {}

for journal, nouns in journal_nouns.items():

    if len(nouns) == 0:
        continue

    # 頻度上位を使う（ノイズ削減）
    freq = Counter(nouns)
    top_words = [w for w, _ in freq.most_common(300)]

    emb = model.encode(top_words)

    journal_vec = np.mean(emb, axis=0)
    journal_embeddings[journal] = journal_vec

    # word-level情報
    for w in top_words:
        word_to_journals[w].add(journal)

# =========================
# 2軸計算
# =========================

results = []

for word, journals in word_to_journals.items():

    # ---- dispersion（分野分布）
    dispersion = len(journals) / len(journal_names)

    # ---- stability（embedding変動）
    vectors = []

    for j in journals:
        if j in journal_embeddings:
            vectors.append(journal_embeddings[j])

    if len(vectors) < 2:
        continue

    sim_matrix = cosine_similarity(vectors)

    # 平均類似度（高い＝安定）
    stability = np.mean(sim_matrix)

    results.append({
        "word": word,
        "stability": stability,
        "dispersion": dispersion
    })

df = pd.DataFrame(results)

# =========================
# 正規化
# =========================

df["stability_norm"] = (df["stability"] - df["stability"].min()) / (df["stability"].max() - df["stability"].min())

df["dispersion_norm"] = (df["dispersion"] - df["dispersion"].min()) / (df["dispersion"].max() - df["dispersion"].min())

# =========================
# 分類
# =========================

def classify(row):
    if row["stability_norm"] > 0.6 and row["dispersion_norm"] < 0.3:
        return "topic"
    elif row["stability_norm"] > 0.6 and row["dispersion_norm"] >= 0.3:
        return "method"
    elif row["stability_norm"] <= 0.6 and row["dispersion_norm"] >= 0.3:
        return "polysemous"
    else:
        return "noise"

df["category"] = df.apply(classify, axis=1)

# =========================
# 保存
# =========================

df.to_csv("word_2d_classification.csv", index=False, encoding="utf-8-sig")

print(df["category"].value_counts())
print(df.head())