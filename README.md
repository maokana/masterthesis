以下のコードを収録
1.下処理
対象となる分野の論文本文のテキストの語の頻度を数え、全ての分野で出現する頻出語を抽出

2.メイン処理
：テキストの頻出語のうち分析対象となる語を単語分散表現に変換し、Wikipedia共通空間上に写像する処理
　embedding_and_trans.py
 ※Wikipediaデータは以下のバージョンを使用。
 vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt.gz


・Wikipediaの共通空間
