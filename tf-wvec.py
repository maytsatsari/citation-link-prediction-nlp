import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



abstracts_data = pd.read_csv('abstracts.txt', sep=r'\|--\|', header=None, names=['ID', 'Abstract'], engine='python', on_bad_lines='skip')


abstracts_data['Abstract'] = abstracts_data['Abstract'].fillna('')


tokenized_abstracts = [abstract.lower().split() for abstract in abstracts_data['Abstract']]

#train Word2Vec model
print("Εκπαίδευση Word2Vec...")
w2v_model = Word2Vec(sentences=tokenized_abstracts, vector_size=100, window=5, min_count=2, workers=4)

# TF-IDF Vectorizer
print("Εφαρμογή TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(abstracts_data['Abstract'])

#Word2Vec embeddings 
def get_abstract_embedding(text, model):
    words = text.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  
    return np.mean(word_vectors, axis=0)

#abstracts to embeddings
print("Μετατροπή abstracts σε embeddings...")
abstracts_data['W2V_Embedding'] = abstracts_data['Abstract'].apply(lambda x: get_abstract_embedding(x, w2v_model))


print("Φόρτωση test.txt...")
test_data = pd.read_csv('test.txt', header=None, names=['PaperID1', 'PaperID2'])

#compute similarity 
print("Υπολογισμός cosine similarity...")
similarities = []

for i, row in test_data.iterrows():
    paper1, paper2 = row['PaperID1'], row['PaperID2']
    
    #TF-IDF similarity
    tfidf_sim = cosine_similarity(X_tfidf[paper1], X_tfidf[paper2])[0][0]
    
    #Word2Vec similarity
    vec1 = abstracts_data.loc[abstracts_data['ID'] == paper1, 'W2V_Embedding'].values[0]
    vec2 = abstracts_data.loc[abstracts_data['ID'] == paper2, 'W2V_Embedding'].values[0]
    w2v_sim = cosine_similarity([vec1], [vec2])[0][0]
    
    #Ensemble similarity (weighted average)
    similarity = 0.6 * tfidf_sim + 0.4 * w2v_sim
    similarities.append(similarity)


test_data['Similarity'] = np.clip(similarities, 0, 1) 
test_data['ID'] = test_data.index


print("Αποθήκευση των αποτελεσμάτων στο test_submission_ensemble.csv...")
test_data[['ID', 'Similarity']].to_csv('test_submission_ensemble.csv', index=False)
