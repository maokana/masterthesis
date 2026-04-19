1.下処理
：対象となる分野の論文本文のテキストの語の頻度を数え、全ての分野で出現する頻出語を抽出
alltext_count.py

2.メイン処理
（1）テキストの頻出語のうち分析対象となる語を単語分散表現に変換し、Wikipedia共通空間上に写像する処理
　embedding_and_trans.py
 ※Wikipediaデータは以下のバージョンを使用。
 vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt.gz
（2）Frontiersの対象となる研究分野の説明文の単語を抽出し、分野間の類似度をJaccard係数化する処理
 maintopicbase_jaccard.py
（3）研究分野間で自誌論文が参加/または自誌に他分野が参加した分野グループ（集合）の類似度
 joinmember_jaccard.py
 (4)研究分野間で相互に論文を自誌に掲載する/相手の雑誌に掲載された回数

3.分析
：
 
