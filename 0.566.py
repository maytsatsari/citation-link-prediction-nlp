import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


print("Ανάγνωση και καθαρισμός αρχείων...")

base_path = "/content/drive/MyDrive/NLP/"


abstracts_data = pd.read_csv(base_path + 'abstracts.txt', sep=r'\|--\|', engine='python', header=None, names=['ID', 'Abstract'])
abstracts_data = abstracts_data.dropna()
abstracts_data = abstracts_data[abstracts_data['ID'].astype(str).str.isnumeric()]
abstracts_data = abstracts_data.astype({'ID': 'uint32'})
abstracts_data['Abstract'] = abstracts_data['Abstract'].fillna('')
print(f"Φορτώθηκαν {len(abstracts_data)} περιλήψεις εργασιών.")


edgelist_data = pd.read_csv(base_path + 'edgelist.txt', sep=',', engine='python', header=None, names=['PaperID1', 'PaperID2'])
edgelist_data = edgelist_data.dropna()
edgelist_data = edgelist_data[
    edgelist_data['PaperID1'].astype(str).str.isnumeric() &
    edgelist_data['PaperID2'].astype(str).str.isnumeric()
]
edgelist_data = edgelist_data.astype({'PaperID1': 'uint32', 'PaperID2': 'uint32'})
print(f"Φορτώθηκαν {len(edgelist_data)} συνδέσεις μεταξύ άρθρων.")


authors_data = pd.read_csv(base_path + 'authors.txt', sep=r'\|--\|', engine='python', header=None, names=['PaperID', 'AuthorNames'])
authors_data = authors_data.dropna()
authors_data['AuthorNames'] = authors_data['AuthorNames'].astype(str).str.split(',')
authors_data = authors_data.explode('AuthorNames')
authors_data = authors_data[authors_data['PaperID'].astype(str).str.isnumeric()]
authors_data = authors_data.astype({'PaperID': 'uint32'})
print(f"Φορτώθηκαν {len(authors_data)} εγγραφές συγγραφέων.")

#SBERT

model = SentenceTransformer('all-mpnet-base-v2')



embeddings_df = pd.read_csv(os.path.join(base_path, 'embeddings.csv'))

#string embeddings to numpy arrays
embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

#merge embeddings with abstracts
abstracts_data = abstracts_data.merge(embeddings_df, on="ID", how="inner")

#dictionary of aythors
author_dict = authors_data.groupby('PaperID')['AuthorNames'].agg(set).to_dict()


test_data = pd.read_csv(base_path + 'test.txt', sep=',', header=None, names=['PaperID1', 'PaperID2'])

#cosine similarity + Jaccard Similarity
print("Υπολογισμός cosine similarity και Jaccard similarity...")
citation_set = set(zip(edgelist_data['PaperID1'], edgelist_data['PaperID2']))
similarities = []

def jaccard_similarity(authors1, authors2):
    if not authors1 or not authors2:
        return 0
    return len(authors1 & authors2) / len(authors1 | authors2)

for row in test_data.itertuples(index=False):
    paper1, paper2 = row.PaperID1, row.PaperID2

    vec1 = abstracts_data.loc[abstracts_data['ID'] == paper1, 'Embedding'].values
    vec2 = abstracts_data.loc[abstracts_data['ID'] == paper2, 'Embedding'].values

    if len(vec1) == 0 or len(vec2) == 0:
        similarity = 0
    else:
        similarity = cosine_similarity([vec1[0]], [vec2[0]])[0][0]

    similarity = (similarity + 1) / 2  # scale σε [0,1]

    authors1 = author_dict.get(paper1, set())
    authors2 = author_dict.get(paper2, set())

    similarity += 0.2 * jaccard_similarity(authors1, authors2)  # Προσθήκη Jaccard score

    if (paper1, paper2) in citation_set or (paper2, paper1) in citation_set:
        similarity += 0.2

    similarities.append(min(1.0, max(0.0, similarity)))


print("Αποθήκευση test_submission_sbert_gat.csv...")
output_file = "test_submission_sbert_gat.csv"
test_data['Similarity'] = similarities
test_data['ID'] = np.arange(len(test_data), dtype='uint32')

test_data[['ID', 'Similarity']].to_csv(output_file, index=False, encoding='utf-8')
print(f"Η εκτέλεση ολοκληρώθηκε επιτυχώς!")
