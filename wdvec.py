import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Ανάγνωση abstracts 
print(" Ανάγνωση αρχείου abstracts.txt")
abstracts_data = pd.read_csv('abstracts.txt', sep=r'\|--\|', header=None, names=['ID', 'Abstract'], engine='python', on_bad_lines='skip')
abstracts_data['Abstract'] = abstracts_data['Abstract'].fillna('')

# --- 2. Προετοιμασία για Word2Vec & Doc2Vec 
print(" Επεξεργασία abstracts")
tokenized_abstracts = [text.lower().split() for text in abstracts_data['Abstract']]
tagged_abstracts = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokenized_abstracts)]

# --- 3. Εκπαίδευση μοντέλων 
print(" Εκπαίδευση Word2Vec και Doc2Vec")
w2v_model = Word2Vec(sentences=tokenized_abstracts, vector_size=100, window=5, min_count=2, workers=4)
d2v_model = Doc2Vec(documents=tagged_abstracts, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

# --- 4. Συνάρτηση για embedding συνδυασμού 
def get_combined_embedding(text, w2v_model, d2v_model, index):
    words = text.lower().split()
    word_vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    doc_vec = d2v_model.dv[str(index)]
    word_mean = np.mean(word_vecs, axis=0) if word_vecs else np.zeros(w2v_model.vector_size)
    return np.concatenate((doc_vec, word_mean))

# --- 5. Υπολογισμός embeddings για abstracts 
print(" Υπολογισμός embeddings")
abstracts_data['Embedding'] = [get_combined_embedding(text, w2v_model, d2v_model, i) for i, text in enumerate(abstracts_data['Abstract'])]

# --- 6. Ανάγνωση test set 
print(" Φόρτωση test.txt")
test_data = pd.read_csv('test.txt', header=None, names=['PaperID1', 'PaperID2'])

# --- 7. Υπολογισμός cosine similarity 
print(" Υπολογισμός cosine similarity")
similarities = []

for _, row in test_data.iterrows():
    p1, p2 = row['PaperID1'], row['PaperID2']
    vec1 = abstracts_data.loc[abstracts_data['ID'] == p1, 'Embedding'].values
    vec2 = abstracts_data.loc[abstracts_data['ID'] == p2, 'Embedding'].values

    if len(vec1) == 0 or len(vec2) == 0:
        sim = 0.0
    else:
        sim = cosine_similarity([vec1[0]], [vec2[0]])[0][0]

    similarities.append(sim)

# --- 8. Δημιουργία υποβολής 
print(" Αποθήκευση test_submission_w2v_d2v.csv")
test_data['Similarity'] = np.clip(similarities, 0, 1)
test_data['ID'] = test_data.index
test_data[['ID', 'Similarity']].to_csv('test_submission_w2v_d2v.csv', index=False)
print(" Η εκτέλεση ολοκληρώθηκε επιτυχώς!")
