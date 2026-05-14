import pandas as pd
from collections import Counter
import re
from itertools import combinations

# ============================================
# CSV読み込み
# ============================================

df = pd.read_csv('researchtopic.csv', encoding='utf-8')

# 新フォーマット:
# A: researchtopic_name
# B: researchtopic_id
# C: description
# D: owner_journal
# E: participant_journal
# F: accepting_flag

# ============================================
# 重複排除
# ============================================
#
# 同一researchtopic_idについて、
# participant_journal違いで複数行存在するため、
# descriptionの重複カウントを防ぐ。
#
# 「owner_journal × researchtopic_id」
# 単位でユニーク化する。
#

unique_topics_df = (
    df[
        [
            'researchtopic_id',
            'description',
            'owner_journal'
        ]
    ]
    .drop_duplicates(
        subset=[
            'researchtopic_id',
            'owner_journal'
        ]
    )
)

# ============================================
# 除外単語
# ============================================

skip_words = set([
    'and','the','of','to','in','a','for','research','is','as','are',
    'on','this','that','or','with','topic','be','by','we','have',
    'from','these','such','not','their','can','has','studies','new',
    'will','but','an','which','been','also','it','other','at','well',
    'between','welcome','how','e','may','s','g','into','t','our',
    'what','all','1','there','both','non','b','must','being','19',
    'its','across','etc','they','ad','early','low','most','2','vr',
    'many','ml','large','about','pd','ar','xr','during','you',
    'older','8226','within','tb','mr','us','ct','5g','2d','ce',
    'gi','i','o','2021','them','while','so','am'
])

# ============================================
# journalごとの頻出単語取得
# ============================================

journal_word_dict = {}

for journal_name, group in unique_topics_df.groupby('owner_journal'):

    # descriptionを結合
    text = ' '.join(
        group['description']
        .fillna('')
        .astype(str)
    ).lower()

    # 単語抽出
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    # stopword除去
    filtered_words = [
        word for word in words
        if word not in skip_words
    ]

    # 単語カウント
    word_counts = Counter(filtered_words)

    # 上位50語取得
    top_50_words = [
        word for word, count
        in word_counts.most_common(50)
    ]

    journal_word_dict[journal_name] = top_50_words

# ============================================
# 頻出語テーブル作成
# 1列目: journal_name
# 2列目以降: top word
# ============================================

word_table_rows = []

for journal, words in journal_word_dict.items():

    row = [journal] + words

    word_table_rows.append(row)

# 列名作成
columns = (
    ['journal_name'] +
    [f'top_word_{i}' for i in range(1, 51)]
)

word_table_df = pd.DataFrame(
    word_table_rows,
    columns=columns
)

# CSV保存
word_table_df.to_csv(
    'rtbase_journal_top_words.csv',
    index=False,
    encoding='utf-8-sig'
)

print("rtbase_journal_top_words.csv を出力しました")

# ============================================
# Jaccard係数計算
# ============================================

def jaccard_index(set1, set2):

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0

    return intersection / union

# 結果格納
jaccard_results = []

journals = list(journal_word_dict.keys())

for journal1, journal2 in combinations(journals, 2):

    set1 = set(journal_word_dict[journal1])
    set2 = set(journal_word_dict[journal2])

    score = jaccard_index(set1, set2)

    jaccard_results.append([
        journal1,
        journal2,
        score
    ])

# DataFrame化
jaccard_df = pd.DataFrame(
    jaccard_results,
    columns=[
        'journal_A',
        'journal_B',
        'jaccard_similarity'
    ]
)

# CSV保存
jaccard_df.to_csv(
    'rtbase_journal_jaccard_similarity.csv',
    index=False,
    encoding='utf-8-sig'
)

print("rtbase_journal_jaccard_similarity.csv を出力しました")
