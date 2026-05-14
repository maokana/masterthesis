import pandas as pd
from collections import defaultdict

# ============================================
# CSV読み込み
# ============================================

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
# ============================================

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

# ============================================
# 前処理
# ============================================

# 欠損除外
df = df.dropna(subset=["owner_journal", "participant_journal"])

# 空白除去
df["owner_journal"] = df["owner_journal"].astype(str).str.strip()
df["participant_journal"] = df["participant_journal"].astype(str).str.strip()

# ============================================
# 各分野のメイントピック数を集計
# ============================================

# research_topic_id単位で重複除去
topic_counts_df = (
    df[["research_topic_id", "owner_journal"]]
    .drop_duplicates()
)

# owner_journalごとのトピック数
topic_counts = (
    topic_counts_df.groupby("owner_journal")
    .size()
    .to_dict()
)

# ============================================
# 分野ペアごとの交流数を集計
# ============================================

# pairごとの双方向交流数
pair_counts = defaultdict(lambda: [0, 0])

for _, row in df.iterrows():

    owner = row["owner_journal"]
    participant = row["participant_journal"]

    # 自己参加除外
    if owner == participant:
        continue

    # 辞書順で統一
    sorted_pair = sorted([owner, participant])

    journal_a = sorted_pair[0]
    journal_b = sorted_pair[1]

    pair_name = f"{journal_a} & {journal_b}"

    # A主催にB参加
    if owner == journal_a:
        pair_counts[pair_name][0] += 1

    # B主催にA参加
    else:
        pair_counts[pair_name][1] += 1

# ============================================
# DataFrame化
# ============================================

results = []

for pair_name, counts in pair_counts.items():

    journal_a, journal_b = pair_name.split(" & ")

    a_topic_count = topic_counts.get(journal_a, 0)
    b_topic_count = topic_counts.get(journal_b, 0)

    a_to_b = counts[0]
    b_to_a = counts[1]

    # 正規化率
    a_to_b_rate = (
        a_to_b / a_topic_count
        if a_topic_count > 0 else 0
    )

    b_to_a_rate = (
        b_to_a / b_topic_count
        if b_topic_count > 0 else 0
    )

    results.append([
        pair_name,
        a_to_b,
        b_to_a,
        a_topic_count,
        b_topic_count,
        a_to_b_rate,
        b_to_a_rate
    ])

# ============================================
# DataFrame作成
# ============================================

result_df = pd.DataFrame(
    results,
    columns=[
        "pair",
        "A_main_to_B_participation_count",
        "B_main_to_A_participation_count",
        "A_main_topic_count",
        "B_main_topic_count",
        "A_to_B_normalized_rate",
        "B_to_A_normalized_rate"
    ]
)

# ソート
result_df = result_df.sort_values("pair")

# ============================================
# CSV出力
# ============================================

result_df.to_csv(
    "pair_interaction_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

print(result_df)

print("pair_interaction_counts.csv を出力しました")
