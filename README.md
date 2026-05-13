# Frontiers論文データ分析

本リポジトリは、論文テキストおよび研究分野間の関係性を分析するためのコード群です。

対象はFrontiersの以下の分野です。

（対象分野）

Computer Science, Psychology, Aging Neuroscience, Applied Mathematics and Statistics, Astronomy and Space Sciences, Big Data, Bioengineering and Biotechnology, Cardiovascular Medicine, Cell and Developmental Biology, Cellular Neuroscience, Cellular and Infection Microbiology, Chemistry, Communication, Digital Health, Earth Science, Ecology and Evolution, Education, Endocrinology, Energy Research, Environmental Science, Forests and Global Change, Genetics, Human Neuroscience, Immunology, Marine Science, Materials, Medicine, Microbiology, Molecular Biosciences, Neurology, Neuroscience, Nutrition, Oncology, Oral Health, Pediatrics, Pharmacology, Physics, Physiology, Plant Science, Political Science, Psychiatry, Public Health, Sociology, Sports and Active Living, Veterinary Science, Virtual Reality, Water,Artificial Intelligence

## ディレクトリ構成

```
main/
├── alltext_count.py --全分野での共通語を抽出するコード
├── embedding_and_trans.py　--単語分散表現及び分野間の単語のcos類似度を求めるコード
├── researchtopicbase_jaccard.py --リサーチトピックの出現単語の一致度をもとに分野間の類似度を求めるコード
├── domainbase_jaccard.py --分野説明の出現単語の一致度をもとに分野間の類似度を求めるコード
├── joinmember_jaccard.py --リサーチトピックの参加メンバー（主催/参加）をもとに分野間の類似度を求めるコード
├── journallist.sql　--リサーチトピックとオーナージャーナル、参加ジャーナルを抽出するSQL
├── mainbase_journallist.py　--上記SQL出力結果を基にオーナージャーナルー参加ジャーナルの形に加工するコード
├── partbase_journallist.py　--上記SQL出力結果を基に参加ジャーナルーオーナージャーナルの形に加工するコード
└── interaction.py --上記SQL出力結果を基に分野間での交流回数をカウントするコード

data/
├── common_words.csv --共通語一覧データ
├── word_similarity.csv --分析対象語の分野間のcos類似度のデータ
├── rtopic_similarity.csv --リサーチトピックの一致度（Jaccard係数）のデータ
├── m_membership_similarity.csv --オーナージャーナルベースでのメンバー一致度（Jaccard係数）のデータ
├── p_membership_similarity.csv --参加ジャーナルベースでのメンバー一致度（Jaccard係数）のデータ
├── interaction_count.csv　--分野間の交流度データ
└── sample_input/　--main内のコードを動かすためのサンプルデータ
    ├── researchtopic.csv
    ├── domain.csv
    ├── journallist.csv
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
  --input small_corpus.zip \ --展開すると分野毎の論文本文のテキストデータが含まれている。
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
python domainbase_jaccard.py \
  --input domain.csv \
  --output topic_similarity.csv
```

### 内容

- 分野説明文の単語抽出
- 単語集合の構築
- Jaccard係数による類似度計算
  
### (3) リサーチトピック説明文ベースの類似度（Jaccard係数）

Frontiersの各研究分野のリサーチトピックの説明文から単語集合を作成し、分野間の類似度を算出する。

### 実行例

```bash
python researchtopicbase_jaccard.py \
  --input researchtopic.csv \
  --output topic_similarity.csv
```

### 内容

- リサーチトピック説明文の単語抽出
- 単語集合の構築
- Jaccard係数による類似度計算
- 
### (4) リサーチトピックのオーナー/参加ジャーナルデータの抽出（SQL）

リサーチトピックのオーナーとなるジャーナルとそのトピックに参加したジャーナルのリストを抽出

https://ma.maonet.org/

```sql
SELECT A.researchtopicid, B.title as main_journal, A.fieldname FROM `researchtopic_fieldjournal` as A left join `researchtopic_ownerjournal` as B on A.researchtopicid = B.researchtopicid;
```

### (5) 分野グループ（参加関係）ベースの類似度

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
  
## 実行環境

```
Python 3.8+
numpy
pandas
gensim
scikit-learn
```

 
