import pandas as pd

# ==========================================
# ファイル読み込み
# ==========================================

# cos類似度ファイル
cos_df = pd.read_csv("cossim.csv")

# 変数ファイル
var_df = pd.read_csv("hensuu.csv")

# ==========================================
# "Source"列をキーとして結合
# ==========================================

merged_df = pd.merge(
    cos_df,
    var_df,
    on="Source",
    how="inner"
)
print(merged_df.columns.tolist())
# ==========================================
# 列名取得
# ==========================================

# cossim側の分析対象語列
cos_columns = [
    col for col in cos_df.columns
    if col != "Source"
]

# hensuu側の変数列
var_columns = [
    col for col in var_df.columns
    if col != "Source"
]

# ==========================================
# 相関係数計算
# ==========================================

results = []

for cos_col in cos_columns:

    row_result = {
        "analysis_word": cos_col
    }

    for var_col in var_columns:

        # Pearson相関
        corr_value = merged_df[cos_col].corr(
            merged_df[var_col]
        )

        row_result[var_col] = corr_value

    results.append(row_result)

# ==========================================
# DataFrame化
# ==========================================

result_df = pd.DataFrame(results)

# ==========================================
# CSV出力
# ==========================================

result_df.to_csv(
    "correlation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(result_df)

print("correlation_results.csv を出力しました")