import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler



abstracts_data = pd.read_csv(
    'abstracts.txt', sep=r'\|--\|', header=None, names=['ID', 'Abstract'],
    engine='python', on_bad_lines='skip'
)
abstracts_data['Abstract'] = abstracts_data['Abstract'].fillna('')

#Sentence-BERT (SBERT) & SPECTER 
sbert_model = SentenceTransformer('all-mpnet-base-v2')
specter_model = SentenceTransformer('allenai/specter')

#embeddings to batch 
sbert_embeddings = sbert_model.encode(abstracts_data['Abstract'].tolist(), show_progress_bar=True)
specter_embeddings = specter_model.encode(abstracts_data['Abstract'].tolist(), show_progress_bar=True)

#NumPy array
abstracts_data['SBERT_Embedding'] = list(sbert_embeddings)
abstracts_data['SPECTER_Embedding'] = list(specter_embeddings)


test_data = pd.read_csv('test.txt', header=None, names=['PaperID1', 'PaperID2'])

#cosine similarity 
abstracts_dict = dict(zip(abstracts_data['ID'], zip(sbert_embeddings, specter_embeddings)))

similarities = []
for i, row in test_data.iterrows():
    paper1, paper2 = row['PaperID1'], row['PaperID2']

    if paper1 in abstracts_dict and paper2 in abstracts_dict:
        vec1_sbert, vec1_specter = abstracts_dict[paper1]
        vec2_sbert, vec2_specter = abstracts_dict[paper2]

        similarity_sbert = cosine_similarity([vec1_sbert], [vec2_sbert])[0][0]
        similarity_specter = cosine_similarity([vec1_specter], [vec2_specter])[0][0]

        similarity = (similarity_sbert + similarity_specter) 
    else:
        similarity = 0 

    similarities.append(similarity)


scaler = MinMaxScaler()
similarities = scaler.fit_transform(np.array(similarities).reshape(-1, 1))


test_data['Similarity'] = similarities
test_data['ID'] = test_data.index  


test_data[['ID', 'Similarity']].to_csv('test_submission_sbert_specter.csv', index=False)

print("Η εκτέλεση ολοκληρώθηκε επιτυχώς!")