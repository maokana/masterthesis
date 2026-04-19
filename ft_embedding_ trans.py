from gensim.models import FastText, KeyedVectors
import numpy as np

def train_fasttext(corpus_path):

    corpus = [line.split() for line in open(corpus_path, encoding='utf-8')]

    model = FastText(
        sentences=corpus,
        vector_size=300,
        window=10,
        min_count=5,
        sg=1,
        workers=4
    )

    return model.wv


def get_vectors(model, vocab):
    return np.array([
        model[w] if w in model else np.zeros(300)
        for w in vocab
    ])


def compare_fasttext(model1, model2, wiki_model, vocab):

    vocab = [w for w in vocab if w in wiki_model]

    X_wiki = np.array([wiki_model[w] for w in vocab])

    X1 = get_vectors(model1, vocab)
    X2 = get_vectors(model2, vocab)

    W1 = np.linalg.lstsq(X1, X_wiki, rcond=None)[0]
    W2 = np.linalg.lstsq(X2, X_wiki, rcond=None)[0]

    def cos(v1, v2):
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 0 if d == 0 else np.dot(v1, v2) / d

    vec1 = X1 @ W1
    vec2 = X2 @ W2

    return [cos(a, b) for a, b in zip(vec1, vec2)]