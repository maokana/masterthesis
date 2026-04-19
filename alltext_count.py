# フォルダ内の全てのテキストファイルをまとめ、単語数を出現回数順にカウントするコード

import glob
import collections

# フォルダ内のすべてのテキストファイルからテキストを読み込む
files = glob.glob("folder/*.txt")

# 全てのテキストファイルの単語の出現回数をカウントする
word_count = collections.Counter()
for file in files:
    with open(file, 'r') as f:
        text = f.read()
        words = text.split()
        word_count.update(words)

# 出現回数が多い順に単語をソートし、出力する
for word, count in word_count.most_common(1000):
    print(word, count)