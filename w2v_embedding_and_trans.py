#!/usr/bin/python3
# ライブラリのインポート
from gensim.models import word2vec
import logging
import os
import tensorflow as tf
import numpy as np
import pickle
import itertools
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='test.log')
tf.compat.v1.disable_eager_execution()

# dp(ディレクトリ)内のファイルの名前を取得しリスト化する関数
def list_file(dp):
  fp_list = []
  for fp in os.listdir(dp):
    fp_list.append(os.path.join(dp, fp))
  fp_list.sort()
  return fp_list

# バッチ処理関数（Adamオプティマイザでの学習用）
 '''
  https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data/40995666
  Return a total of `num` random samples and labels. 
  '''
def next_batch(num, data, labels):
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# 各研究分野毎に学習された単語分散表現をWikipedia言語空間に線形写像する変換行列を生成する関数
def learn_linear_transformation(vec_map_from, vec_map_to):
  # ベクトル空間次元数
  _, dvec_map_to = vec_map_to.shape
  _, dvec_map_from = vec_map_from.shape

  # 写像元
  x = tf.compat.v1.placeholder(tf.float32, [None, dvec_map_from])

  # 写像先
  z = tf.compat.v1.placeholder(tf.float32, [None, dvec_map_to])

  # 変換行列（学習前初期値:標準偏差 0.01 のガウス分布）
  W = tf.Variable(tf.compat.v1.random_normal([dvec_map_from, dvec_map_to], stddev=0.01))

  # 損失関数定義 $L = \sum_{i=1}^n || Wx_i - z_i ||^2$
  loss = tf.reduce_sum(tf.square(z - tf.matmul(x, tf.transpose(W))))

  # Adam オプティマイザを確率的勾配降下法に使います.
  train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    # バッチサイズ 100 の確率的勾配降下法を10000回行う
    for i in range(10000000):
      batch_x, batch_z = next_batch(100, vec_map_from, vec_map_to)
      sess.run(train_step, feed_dict={x: batch_x, z:batch_z})
      #書き直し
      # less = sess.run(train_step, feed_dict={x: batch_x, z:batch_z})
      if i % 1000 == 0:
        print('Step: %d, Loss: %f'% (i, sess.run(loss, feed_dict={x: batch_x, z:batch_z})))
      #書き直し
      # print('Step: %d, Loss: %f'% (i, less))

    # 学習した変換行列を抽出
    W_ = sess.run([W])

  #print(W_[0])
  return W_

# 出力されたリストデータ（lst)を指定したファイルパス（fp）にpickle形式で保存する関数
def savepickle(lst, fp):
  with open(fp, 'wb') as f:
    pickle.dump(lst, f)

# 保存したpickle形式のファイルをロードする関数
def loadpickle(fp):
  with open(fp, 'rb') as f:
    return pickle.load(f)

# 単語ベクトル間のコサイン類似度を計算する関数
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#2つの異なる空間にある同一の単語(例えばanalysis)について、一方の単語の値を他方の空間に線形変換し写像する
#以下、メイン処理
def main():
  #src_txt_listは下記のようなデータが生成されるはず
  #src_txt_list = ['Frontiers_in_hoge.txt', 'Frontiers_in_huga.txt', 'Frontiers_in_humu.txt']
  dp_tgt_vec = 'tgt_vec'
  dp_tgt_bin = 'tgt_bin'
  fn_tgt_vec = 'vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt'
  dp_src_txt = 'src_txt'
  dp_src_bin = 'src_bin'
  dp_pickle = 'pickle'
  dp_cossim = 'cossim'

  cos_sim_result_list = []

  #各テキストの単語の単語分散表現を取得
  src_txt_list = list_file(dp_src_txt)
  for fp_txt in src_txt_list:
    fn_txt = os.path.basename(fp_txt)
    fn_bin = '{}.bin'.format(os.path.splitext(fn_txt)[0])
    fp_bin = os.path.join(dp_src_bin, fn_bin)
    logging.info('Start learning process for "{}"'.format(fn_txt))
    if os.path.isfile(fp_bin):
      logging.info('"{}" already exits. Skip learning "{}"'.format(fn_bin, fn_txt))
    else:
      logging.info('Start learning "{}"'.format(fn_txt))
      corpus = word2vec.Text8Corpus(fp_txt)
      model = word2vec.Word2Vec(corpus, min_count=1, vector_size=300) 
      #min_count:n個以上の出現単語を対象とする、vector_size:次元数(wikiが300dimのため300指定)
      model.wv.save_word2vec_format(fp_bin, binary=True)
      logging.info('Finished learning "{}"'.format(fn_txt))

  #線形写像の軸となる辞書(vocab_dic)を設定する
  #key:変換先（wiki)のcorpusでの単語、value:変換前（各分野）のcorpusでの単語
  #vocab_src（wiki)はkey,vocab_tgt(各分野）はvalueを設定
  #vocab_dicを変更したらpickleも再生成が必要
  #vocab_dic = {'may':'may', 'also':'also', 'non':'non', 'however':'however', 'could':'could', 'al':'al', 'et':'et'}--修論時点
  vocab_dic = {'may':'may', 'also':'also', 'non':'non', 'however':'however', 'could':'could', 'al':'al', 'et':'et','the':'the','and':'and','of':'of','with':'with','for':'for','that':'that','in':'in','this':'this','from':'from'}
  vocab_src = vocab_dic.keys()
  vocab_tgt = vocab_dic.values()

  logging.info('Start linear transformation process')

  is_model_tgt_loaded = False
  #高速化のためwikipediaのvector textをbinaryに変換
  fp_tgt_vec = os.path.join(dp_tgt_vec, fn_tgt_vec)
  fn_tgt_bin = '{}.bin'.format(os.path.splitext(fn_tgt_vec)[0])
  fp_tgt_bin = os.path.join(dp_tgt_bin, fn_tgt_bin)
  #高速化のためbinに変換済みであれば変換処理を飛ばす
  if os.path.isfile(fp_tgt_bin):
    logging.info('"{}" already exists. Skip converting vector formatted text to binary file.'.format(fn_tgt_bin))
  else:
    logging.info('"{}" does not exists. Start converting vector formatted text to binary file.'.format(fn_tgt_bin))
    model_tgt = word2vec.KeyedVectors.load_word2vec_format(fp_tgt_vec, binary=False)
    model_tgt.save_word2vec_format(fp_tgt_bin, binary=True)
    is_model_tgt_loaded = True

  src_txt_list = list_file(dp_src_txt)
  for i,fp_txt in enumerate(src_txt_list):
    fn_txt = os.path.basename(fp_txt)
    fn_bin = '{}.bin'.format(os.path.splitext(fn_txt)[0])
    fp_bin = os.path.join(dp_src_bin, fn_bin)
    fn_pickle = '{}.pickle'.format(os.path.splitext(fn_txt)[0])
    fp_pickle = os.path.join(dp_pickle, fn_pickle)

    #高速化のためpickleに変換済みであれば変換処理を飛ばす
    if os.path.isfile(fp_pickle):
      pass
    else:
      if not is_model_tgt_loaded:
        logging.info('Loading "{}"'.format(fn_tgt_bin))
        model_tgt = word2vec.KeyedVectors.load_word2vec_format(fp_tgt_bin, binary=True)
        vec_tgt = model_tgt[vocab_src]
        is_model_tgt_loaded = True

      logging.info('({}/{}) Learning linear transformation "{}"'.format(i+1, len(src_txt_list), fn_bin))
      model_src = word2vec.KeyedVectors.load_word2vec_format(fp_bin, binary=True)

      # 線形変換を行う変換行列を生成
      try:
        vec_src = model_src[vocab_tgt]
        W = learn_linear_transformation(vec_src, vec_tgt)
        savepickle(W, fp_pickle)
      except KeyError as e:
        logging.error('vocab_dic error: {} in "{}"'.format(e, fn_bin))


  # 変換用行列を基に、analyse_listの分析ターゲットの単語ベクトルを、元の行列から写像先へのベクトルへと変換する
  # 分析ターゲットの単語リストを指定
  analyse_list = ['analysis','model','design','figure','describe','base','measure','range','estimate','factor','significant','found','lead','term','make','method','parameter','represent','state','area','current','interest','approach','point','available','work']
  # 2024/6/2変更
  # analyse_list =['analysis','model','design','figure','describe','based','measure','range','estimate','factor','significant','found','lead','terms','make','method','parameter','represent','state','area','current','interest','approach','point','available','work','results','effect','related','system','mechanisms','cognitive','cancer','measured','hospital','impact','recent','calculated','sequence']
  src_txt_list = list_file(dp_src_txt)
  fp_bin_1_past = None
  fp_bin_2_past = None
  #分野間でのコサイン類似度計算のために、pickleファイルで総当りの組み合わせを生成する
  fp_combi_list = list(itertools.combinations(src_txt_list, 2))
  
  #計算結果を保存したファイルを指定したフォルダに出力
  for i, fp_combi in enumerate(fp_combi_list):
    fn_txt_1 = os.path.basename(fp_combi[0])
    fn_txt_2 = os.path.basename(fp_combi[1])
    fn_bin_1 = '{}.bin'.format(os.path.splitext(fn_txt_1)[0])
    fp_bin_1 = os.path.join(dp_src_bin, fn_bin_1)
    fn_bin_2 = '{}.bin'.format(os.path.splitext(fn_txt_2)[0])
    fp_bin_2 = os.path.join(dp_src_bin, fn_bin_2)
    fn_pickle_1 = '{}.pickle'.format(os.path.splitext(fn_txt_1)[0])
    fp_pickle_1 = os.path.join(dp_pickle, fn_pickle_1)
    fn_pickle_2 = '{}.pickle'.format(os.path.splitext(fn_txt_2)[0])
    fp_pickle_2 = os.path.join(dp_pickle, fn_pickle_2)
    fn_cossim = '{}_vs_{}.pickle'.format(fn_txt_1, fn_txt_2)
    fp_cossim = os.path.join(dp_cossim, fn_cossim)

    logging.info('({}/{}) {} vs {}'.format(i+1, len(fp_combi_list), fn_txt_1, fn_txt_2))

    if os.path.isfile(fp_cossim):
      logging.info('"{}" already exists. Skip cos simulation.'.format(fn_cossim))
      cos_sim_result_list.append(loadpickle(fp_cossim))
    else:
      if fp_bin_1 == fp_bin_1_past:
        logging.info('"{}" already loaded. Skip loading.'.format(fn_bin_1))
      else:
        model_1 = word2vec.KeyedVectors.load_word2vec_format(fp_bin_1, binary=True)
        fp_bin_1_past = fp_bin_1
      if fp_bin_2 == fp_bin_2_past:
        logging.info('"{}" already loaded. Skip loading.'.format(fn_bin_2))
      else:
        model_2 = word2vec.KeyedVectors.load_word2vec_format(fp_bin_2, binary=True)
        fp_bin_2_past = fp_bin_2

      try:
        W_1 = loadpickle(fp_pickle_1)
      except FileNotFoundError as e:
        logging.error('"{}" not found. Skip'.format(fn_pickle_1))
        continue
      try:
        W_2 = loadpickle(fp_pickle_2)
      except FileNotFoundError as e:
        logging.error('"{}" not found. Skip'.format(fn_pickle_2))
        continue

      try:
        model_1_list = model_1[analyse_list]
      except KeyError as e:
        logging.error('analyse_list error: {} in "{}". Skip cos simulation.'.format(e, fp_bin_1))
        continue
      try:
        model_2_list = model_2[analyse_list]
      except KeyError as e:
        logging.error('analyse_list error: {} in "{}". Skip cos simulation.'.format(e, fp_bin_2))
        continue
      #分野毎にanalyse_listの単語をWikipedia言語空間に写像した単語ベクトル値を取得
      model_1_list2 = [np.dot(i,W_1[0]) for i in model_1_list]
      model_2_list2 = [np.dot(i,W_2[0]) for i in model_2_list]

      #2分野間でのコサイン類似度の計算
      cos_sim_list = []
      for i, v in zip(model_1_list2, model_2_list2):
        cos_sim_list.append(cos_sim(i,v))
      cos_sim_list.insert(0, '{} vs {}'.format(fn_txt_1, fn_txt_2))
      cos_sim_result_list.append(cos_sim_list)
      savepickle(cos_sim_list, fp_cossim)

  header = analyse_list
  header.insert(0, 'Source')
  cos_sim_result_list.insert(0, header)

  df = pd.DataFrame(cos_sim_result_list)
  print(df)
  df.to_csv('cossim.csv')

if __name__ == '__main__':
  main()

