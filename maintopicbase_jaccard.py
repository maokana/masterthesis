import pandas as pd

def read_data(file_path):
    # CSVファイルを行ごとに読み込む（列数が異なっていても問題ない）
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    categories = {}
    
    for line in lines:
        # 行をカンマで分割
        columns = line.strip().split(',')
        category = columns[0]  # カテゴリー番号（最初の列）
        fields = set(columns[1:])  # 分野（残りの列）
        
        # 空の分野を除外
        fields = {field for field in fields if field.strip()}
        
        if category not in categories:
            categories[category] = set()
        categories[category].update(fields)
    
    return categories

def jaccard_index(set1, set2):
    # Jaccard係数を計算
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def compute_jaccard_for_categories(categories):
    # カテゴリー間のJaccard係数を計算
    results = {}
    category_list = list(categories.keys())
    for i in range(len(category_list)):
        for j in range(i + 1, len(category_list)):
            cat1 = category_list[i]
            cat2 = category_list[j]
            jaccard = jaccard_index(categories[cat1], categories[cat2])
            results[(cat1, cat2)] = jaccard
    return results

# 実行例
file_path = 'jaccard_sample.csv'  # CSVファイルのパスを指定
categories = read_data(file_path)
jaccard_results = compute_jaccard_for_categories(categories)

# 結果表示
for (cat1, cat2), jaccard in jaccard_results.items():
    print(f"カテゴリー {cat1} と カテゴリー {cat2} のJaccard係数: {jaccard}")
