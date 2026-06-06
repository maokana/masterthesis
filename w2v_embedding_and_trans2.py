#analyse_listの単語が全分野のファイルに出現することが条件（その部分の回避策はとっていない）
#!/usr/bin/python3
from gensim.models import word2vec, KeyedVectors
import logging
import os
import numpy as np
import pickle
import itertools
import pandas as pd

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename='test.log'
)


# ===== ファイル一覧（ディレクトリ除外）=====
def list_file(dp):
    return sorted([
        os.path.join(dp, f)
        for f in os.listdir(dp)
        if os.path.isfile(os.path.join(dp, f))
    ])


# ===== 線形変換 =====
def learn_linear_transformation(X, Z):
    W, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return W


def savepickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def loadpickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def cos_sim(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom


def main():

    dp_tgt_vec = 'tgt_vec'
    dp_tgt_bin = 'tgt_bin'
    fn_tgt_vec = 'vec_enwiki-20160601_w2v_min50_win10_dim300_skipgram_ns5.txt'

    dp_src_txt = 'src_txt'
    dp_src_bin = 'src_bin'
    dp_pickle = 'pickle'

    os.makedirs(dp_src_bin, exist_ok=True)
    os.makedirs(dp_tgt_bin, exist_ok=True)
    os.makedirs(dp_pickle, exist_ok=True)

    # ===== Word2Vec学習 =====
    src_txt_list = list_file(dp_src_txt)

    for fp_txt in src_txt_list:
        fn_txt = os.path.basename(fp_txt)
        fp_bin = os.path.join(dp_src_bin, fn_txt + ".bin")

        if not os.path.exists(fp_bin):
            logging.info(f"Training {fn_txt}")
            corpus = word2vec.Text8Corpus(fp_txt)
            model = word2vec.Word2Vec(corpus, min_count=1, vector_size=300)
            model.wv.save_word2vec_format(fp_bin, binary=True)

    # ===== vocab_dic（完全保持）=====
    vocab_dic = {
        'few':'few','many':'many','less':'less','or':'or','but':'but','if':'if','while':'while',
        'because':'because','although':'although','therefore':'therefore','thus':'thus',
        'however':'however','all':'all','both':'both','each':'each','some':'some','any':'any',
        'more':'more','most':'most','between':'between','into':'into','among':'among',
        'within':'within','after':'after','before':'before','over':'over','under':'under',
        'through':'through','across':'across','during':'during'
    }

    # ===== Wikipediaベクトル =====
    fp_tgt_vec = os.path.join(dp_tgt_vec, fn_tgt_vec)
    fp_tgt_bin = os.path.join(dp_tgt_bin, fn_tgt_vec + ".bin")

    if not os.path.exists(fp_tgt_bin):
        logging.info("Converting wiki vec -> bin")
        model_tgt = KeyedVectors.load_word2vec_format(fp_tgt_vec, binary=False)
        model_tgt.save_word2vec_format(fp_tgt_bin, binary=True)

    model_tgt = KeyedVectors.load_word2vec_format(fp_tgt_bin, binary=True)

    # ===== 線形変換 =====
    for fp_txt in src_txt_list:

        fn_txt = os.path.basename(fp_txt)
        fp_bin = os.path.join(dp_src_bin, fn_txt + ".bin")
        fp_pickle = os.path.join(dp_pickle, fn_txt + ".pickle")

        if os.path.exists(fp_pickle):
            continue

        try:
            model_src = KeyedVectors.load_word2vec_format(fp_bin, binary=True)
        except Exception as e:
            logging.error(f"Missing bin {fn_txt}: {e}")
            continue

        # 共通語彙
        common_pairs = [
            (s, t) for s, t in vocab_dic.items()
            if s in model_tgt and t in model_src
        ]

        if len(common_pairs) < 5:
            logging.error(f"Too few vocab in {fn_txt}")
            continue

        vocab_src = [t for _, t in common_pairs]
        vocab_tgt = [s for s, _ in common_pairs]

        vec_src = model_src[vocab_src]
        vec_tgt = model_tgt[vocab_tgt]

        W = learn_linear_transformation(vec_src, vec_tgt)
        savepickle(W, fp_pickle)

    # ===== analyse_list（完全保持）=====
    analyse_list = ['figure','study','analysis','studies','results','model','based','research','effect','associated','compared','related','effects','observed','system','development','factors','function','models','process','work','method','performance','methods','factor','findings','target','figures','described','signaling','learning','considered','analyzed','case','structure','evidence','functional','analyses','impact','result','mechanisms','interaction','cognitive','standard','experimental','revealed','systems','features','ability','design','formation','regulation','mechanism','flow','derived','influence','condition','detected','developed','determined','association','comparison','form','effective','functions','processing','subjects','defined','corresponding','proposed','interactions','consistent','limited','exposure','understanding','measures','properties','affect','resulting','suggested','assessment','components','patterns','stimulation','sequences','context','determine','solution','inhibition','contributions','investigated','attention','suggesting','assessed','image','application','caused','generated','imaging','affected','combined','pattern','baseline','availability','contributed','intervention']
    # ===== コサイン類似度 =====
    results = []
    fp_combi_list = list(itertools.combinations(src_txt_list, 2))

    for fp1, fp2 in fp_combi_list:

        fn1 = os.path.basename(fp1)
        fn2 = os.path.basename(fp2)

        try:
            model1 = KeyedVectors.load_word2vec_format(
                os.path.join(dp_src_bin, fn1 + ".bin"), binary=True
            )
            model2 = KeyedVectors.load_word2vec_format(
                os.path.join(dp_src_bin, fn2 + ".bin"), binary=True
            )

            W1 = loadpickle(os.path.join(dp_pickle, fn1 + ".pickle"))
            W2 = loadpickle(os.path.join(dp_pickle, fn2 + ".pickle"))

            words1 = [w for w in analyse_list if w in model1]
            words2 = [w for w in analyse_list if w in model2]

            if len(words1) == 0 or len(words2) == 0:
                continue

            v1 = model1[words1]
            v2 = model2[words2]

        except Exception as e:
            logging.error(f"Skip {fn1}-{fn2}: {e}")
            continue

        v1 = np.array([np.dot(x, W1) for x in v1])
        v2 = np.array([np.dot(x, W2) for x in v2])

        sims = [cos_sim(a, b) for a, b in zip(v1, v2)]
        sims.insert(0, f"{fn1} vs {fn2}")
        results.append(sims)

    header = ['Source'] + analyse_list
    results.insert(0, header)

    pd.DataFrame(results).to_csv("cossim.csv", index=False)

    print("Finished successfully")


if __name__ == "__main__":
    main()
