import pandas as pd
from collections import defaultdict

# ============================================
# CSV読み込み
# ============================================

# 同じディレクトリに journallist.csv がある前提
df = pd.read_csv("journallist.csv")

# 想定列:
# research_topic_id
# owner_journal
# participant_journal

# ============================================
# 分野ペアごとの交流数を集計
# ============================================

# 結果格納用
pair_counts = defaultdict(lambda: [0, 0])

# 1行ずつ処理
for _, row in df.iterrows():

    owner = str(row["owner_journal"]).strip()
    participant = str(row["participant_journal"]).strip()

    # 自分自身は除外したい場合
    if owner == participant:
        continue

    # ペア名を辞書順で統一
    pair_name = ''.join(sorted([owner, participant]))

    # pair_name の先頭側を基準に方向を固定
    sorted_pair = sorted([owner, participant])

    # owner が pair の先頭なら index 0
    if owner == sorted_pair[0]:
        pair_counts[pair_name][0] += 1

    # owner が pair の後ろなら index 1
    else:
        pair_counts[pair_name][1] += 1

# ============================================
# DataFrame化
# ============================================

results = []

for pair_name, counts in pair_counts.items():

    results.append([
        pair_name,
        counts[0],
        counts[1]
    ])

result_df = pd.DataFrame(
    results,
    columns=[
        "pair",
        "forward_count",
        "reverse_count"
    ]
)

# pair列でソート
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