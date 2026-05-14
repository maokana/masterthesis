import pandas as pd
import itertools

# ==========================================
# 入力ファイル読み込み
# ==========================================
# ファイル名:
# researchtopics_v2.csv
#
# 列定義:
# A: research_topic_name
# B: research_topic_id
# C: description
# D: owner_journal
# E: participant_journal
# F: submission_flag
# ==========================================

df = pd.read_csv("researchtopics_v2.csv")

# 列名定義
df.columns = [
    "research_topic_name",
    "research_topic_id",
    "description",
    "owner_journal",
    "participant_journal",
    "submission_flag"
]

# ==========================================
# 前処理
# ==========================================

# owner / participant が空の行を除外
df = df.dropna(subset=["owner_journal", "participant_journal"])

# ==========================================
# （1）
# owner_journal ごとに
# participant_journal を横持ち変換
# ==========================================

owner_group = (
    df.groupby("owner_journal")["participant_journal"]
      .apply(list)
      .reset_index()
)

# リストを列展開
participant_cols = pd.DataFrame(
    owner_group["participant_journal"].tolist()
)

# 列名設定
participant_cols.columns = [
    f"participant_journal_{i+1}"
    for i in range(participant_cols.shape[1])
]

# 結合
owner_output = pd.concat(
    [owner_group["owner_journal"], participant_cols],
    axis=1
)

# CSV出力
owner_output.to_csv(
    "mainjbase_journallist.csv",
    index=False,
    encoding="utf-8-sig"
)

print("mainjbase_journallist.csv を出力しました")


# ==========================================
# （2）
# participant_journal ごとに
# owner_journal を横持ち変換
# ==========================================

participant_group = (
    df.groupby("participant_journal")["owner_journal"]
      .apply(list)
      .reset_index()
)

# リストを列展開
owner_cols = pd.DataFrame(
    participant_group["owner_journal"].tolist()
)

# 列名設定
owner_cols.columns = [
    f"owner_journal_{i+1}"
    for i in range(owner_cols.shape[1])
]

# 結合
participant_output = pd.concat(
    [participant_group["participant_journal"], owner_cols],
    axis=1
)

# CSV出力
participant_output.to_csv(
    "partjbase_journallist.csv",
    index=False,
    encoding="utf-8-sig"
)

print("partjbase_journallist.csv を出力しました")


# ==========================================
# Jaccard係数計算関数
# ==========================================

def jaccard_similarity(set_a, set_b):

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    if len(union) == 0:
        return 1.0

    return len(intersection) / len(union)


# ==========================================
# （3）
# owner_journal 同士の
# participant_journal 一致度
# ==========================================

# owner_journal -> participant集合
owner_dict = (
    df.groupby("owner_journal")["participant_journal"]
      .apply(set)
      .to_dict()
)

owner_pairs = list(
    itertools.combinations(sorted(owner_dict.keys()), 2)
)

owner_jaccard_results = []

for journal_a, journal_b in owner_pairs:

    set_a = owner_dict[journal_a]
    set_b = owner_dict[journal_b]

    score = jaccard_similarity(set_a, set_b)

    owner_jaccard_results.append({
        "journal_a": journal_a,
        "journal_b": journal_b,
        "jaccard_coefficient": score
    })

owner_jaccard_df = pd.DataFrame(owner_jaccard_results)

# CSV出力
owner_jaccard_df.to_csv(
    "owner_journal_jaccard.csv",
    index=False,
    encoding="utf-8-sig"
)

print("owner_journal_jaccard.csv を出力しました")


# ==========================================
# （4）
# participant_journal 同士の
# owner_journal 一致度
# ==========================================

# participant_journal -> owner集合
participant_dict = (
    df.groupby("participant_journal")["owner_journal"]
      .apply(set)
      .to_dict()
)

participant_pairs = list(
    itertools.combinations(sorted(participant_dict.keys()), 2)
)

participant_jaccard_results = []

for journal_a, journal_b in participant_pairs:

    set_a = participant_dict[journal_a]
    set_b = participant_dict[journal_b]

    score = jaccard_similarity(set_a, set_b)

    participant_jaccard_results.append({
        "journal_a": journal_a,
        "journal_b": journal_b,
        "jaccard_coefficient": score
    })

participant_jaccard_df = pd.DataFrame(
    participant_jaccard_results
)

# CSV出力
participant_jaccard_df.to_csv(
    "participant_journal_jaccard.csv",
    index=False,
    encoding="utf-8-sig"
)

print("participant_journal_jaccard.csv を出力しました")


print("すべての処理が完了しました")
