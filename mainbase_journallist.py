import pandas as pd

# CSV読み込み
# 同じディレクトリに rtopic_main_part_pair.csv がある前提
df = pd.read_csv("rtopic_main_part_pair.csv")

# 列名例:
# research_topic_id
# owner_journal
# participant_journal

# ----------------------------------------
# オーナージャーナルごとに
# 参加ジャーナルを横持ちに変換
# ----------------------------------------

result = (
    df.groupby("owner_journal")["participant_journal"]
      .apply(list)
      .reset_index()
)

# リストを列展開
participants = pd.DataFrame(
    result["participant_journal"].tolist()
)

# 列名を設定
participants.columns = [
    f"participant_journal_{i+1}"
    for i in range(participants.shape[1])
]

# オーナージャーナル列と結合
final_df = pd.concat(
    [result["owner_journal"], participants],
    axis=1
)

# 出力確認
print(final_df)

# CSV保存
final_df.to_csv("mainjbase_journallist.csv", index=False)