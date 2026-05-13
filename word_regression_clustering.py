!pip install pandas numpy scikit-learn umap-learn matplotlib seaborn

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CSV読み込み
# ============================================

# 同じディレクトリにある前提
df = pd.read_csv("input.csv")

# 想定列:
# word
# pair
# cos_similarity
# topic_similarity
# domain_similarity
# member_similarity
# interaction_count

# ============================================
# 単語ごとに回帰係数を推定
# ============================================

results = []

words = df["word"].unique()

for word in words:

    sub = df[df["word"] == word]

    # 説明変数
    X = sub[[
        "topic_similarity",
        "domain_similarity",
        "member_similarity",
        "interaction_count"
    ]]

    # 目的変数
    y = sub["cos_similarity"]

    # サンプル不足回避
    if len(sub) < 5:
        continue

    # 回帰
    model = LinearRegression()
    model.fit(X, y)

    results.append([
        word,
        model.coef_[0],
        model.coef_[1],
        model.coef_[2],
        model.coef_[3]
    ])

# DataFrame化
coef_df = pd.DataFrame(
    results,
    columns=[
        "word",
        "beta_topic",
        "beta_domain",
        "beta_member",
        "beta_interaction"
    ]
)

# 保存
coef_df.to_csv(
    "word_regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)

print("word_regression_coefficients.csv を出力しました")

# ============================================
# クラスタリング
# ============================================

features = coef_df[[
    "beta_topic",
    "beta_domain",
    "beta_member",
    "beta_interaction"
]]

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ============================================
# KMeans
# ============================================

# クラスタ数（調整可能）
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

coef_df["cluster"] = clusters

# ============================================
# UMAPで2次元化
# ============================================

reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.2,
    random_state=42
)

embedding = reducer.fit_transform(X_scaled)

coef_df["umap_x"] = embedding[:, 0]
coef_df["umap_y"] = embedding[:, 1]

# ============================================
# CSV保存
# ============================================

coef_df.to_csv(
    "word_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

print("word_clusters.csv を出力しました")

# ============================================
# 可視化
# ============================================

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=coef_df,
    x="umap_x",
    y="umap_y",
    hue="cluster",
    palette="tab10",
    s=100
)

# 単語ラベル表示
for _, row in coef_df.iterrows():

    plt.text(
        row["umap_x"],
        row["umap_y"],
        row["word"],
        fontsize=8
    )

plt.title("Word Clustering based on Regression Coefficients")

plt.tight_layout()

# 保存
plt.savefig(
    "word_cluster_map.png",
    dpi=300
)

print("word_cluster_map.png を出力しました")