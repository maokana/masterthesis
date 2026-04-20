# Frontiers論文データ分析

本リポジトリは、論文テキストおよび研究分野間の関係性を分析するためのコード群です。

## ディレクトリ構成

```
main/
├── alltext_count.py
├── embedding_and_trans.py
├── maintopicbase_jaccard.py
├── joinmember_jaccard.py
└── hogehoge.sql
data/
├── common_words.csv
├── word_similarity.csv
├── topic_similarity.csv
├── membership_similarity.csv
├── interaction_count.csv
└── sample_corpus.zip
    └── corpus/
        ├── Medicine.txt
        ├── Biology.txt
        ├── Physics.txt
        └── Chemistry.txt
db/
├── schema.sql
├── sample_insert.sql   
└── sample_input.json
```

## 1. 下処理

対象分野の論文本文テキストから語の頻度を集計し、全分野で共通して出現する頻出語を抽出する。

### 実行例

```bash
python alltext_count.py --input_dir ./texts --output common_words.csv
```

### 内容

- 論文テキストの読み込み
- トークン化・前処理
- 単語頻度の計算
- 分野横断での共通頻出語抽出

## 2. メイン処理

### (1) 単語分散表現 + Wikipedia空間への写像

頻出語のうち分析対象語をベクトル化し、Wikipedia共通空間に写像する。

### 実行例

```bash
python embedding_and_trans.py \
  --input common_words.csv \
  --w2v vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt.gz \
  --output embeddings.npy
```

### 使用データ

```
vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt.gz
```

### 内容

- 単語のWord2Vec表現取得
- Wikipediaベクトル空間へのマッピング
- 分析用ベクトルの生成

### (2) 分野説明文ベースの類似度（Jaccard係数）

Frontiersの各研究分野の説明文から単語集合を作成し、分野間の類似度を算出する。

### 実行例

```bash
python maintopicbase_jaccard.py \
  --input topics.txt \
  --output topic_similarity.csv
```

### 内容

- 分野説明文の単語抽出
- 単語集合の構築
- Jaccard係数による類似度計算

### (3) 分野グループ（参加関係）ベースの類似度

自誌論文の参加関係から分野集合を作成し、類似度を算出する。

### 実行例

```bash
python joinmember_jaccard.py \
  --input membership.csv \
  --output membership_similarity.csv
```

### 内容

- 分野ごとの参加分野集合を構築
- Jaccard係数による分野間類似度の計算

### (4) 分野間の論文掲載関係（SQL）

研究分野間での相互掲載関係を集計する。

```sql
SELECT
    source_field,
    target_field,
    COUNT(*) AS publication_count
FROM
    publications
GROUP BY
    source_field,
    target_field;
```

## 実行環境

```
Python 3.8+
numpy
pandas
gensim
scikit-learn
```
 
