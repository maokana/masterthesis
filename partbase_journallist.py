import pandas as pd

# CSV読み込み
# 同じディレクトリに rtopic_main_part_pair.csv がある前提
df = pd.read_csv("rtopic_main_part_pair.csv")

# 列名例:
# research_topic_id
# owner_journal
# participant_journal

# ----------------------------------------
# participant_journal ごとに
# owner_journal を横持ちに変換
# ----------------------------------------

result = (
    df.groupby("participant_journal")["owner_journal"]
      .apply(list)
      .reset_index()
)

# リストを列展開
owners = pd.DataFrame(
    result["owner_journal"].tolist()
)

# 列名を設定
owners.columns = [
    f"owner_journal_{i+1}"
    for i in range(owners.shape[1])
]

# participant_journal列と結合
final_df = pd.concat(
    [result["participant_journal"], owners],
    axis=1
)

# 出力確認
print(final_df)

# CSV保存
final_df.to_csv("partjbase_journallist.csv", index=False)