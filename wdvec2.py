import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


abstracts = pd.read_csv('abstracts.txt', sep=r'\|--\|', header=None, names=['ID', 'Abstract'], engine='python')
abstracts['Abstract'] = abstracts['Abstract'].fillna('')

#Word2Vec - Doc2Vec 
texts = [text.lower().split() for text in abstracts['Abstract']]
tagged = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(texts)]


w2v = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=2, workers=2)
d2v = Doc2Vec(tagged, vector_size=100, window=5, min_count=2, epochs=20, workers=2)

#embeddings
def embed(text, idx):
    words = [w for w in text.lower().split() if w in w2v.wv]
    vec1 = np.mean([w2v.wv[w] for w in words], axis=0) if words else np.zeros(w2v.vector_size)
    vec2 = d2v.dv[str(idx)] if str(idx) in d2v.dv else np.zeros(d2v.vector_size)
    return np.concatenate([vec1, vec2])

X = np.array([embed(text, i) for i, text in enumerate(abstracts['Abstract'])])

# PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)


test = pd.read_csv('test.txt', header=None, names=['PaperID1', 'PaperID2'])

#compute distances- features 
def extract_features(idx1, idx2):
    if idx1 >= len(X_reduced) or idx2 >= len(X_reduced):
        return [0, 0, 0]
    v1, v2 = X_reduced[idx1], X_reduced[idx2]
    return [
        cosine_similarity([v1], [v2])[0][0],
        euclidean_distances([v1], [v2])[0][0],
        manhattan_distances([v1], [v2])[0][0]
    ]

features = np.array([extract_features(i, j) for i, j in zip(test['PaperID1'], test['PaperID2'])])
features = MinMaxScaler().fit_transform(features)


labels = np.random.randint(0, 2, len(features))

#XGBoost
X_tr, X_val, y_tr, y_val = train_test_split(features, labels, test_size=0.2)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric="logloss")
model.fit(X_tr, y_tr)


test['Similarity'] = model.predict_proba(features)[:, 1]


test['ID'] = test.index
test[['ID', 'Similarity']].to_csv('test_submission_xgb.csv', index=False)
print("Η υποβολή αποθηκεύτηκε.")
