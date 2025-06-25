import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.spatial.distance import cosine
from fuzzywuzzy import fuzz


base_path = "/content/drive/MyDrive/NLP/"


abstracts = pd.read_csv(base_path + 'abstracts.txt', sep=r'\|--\|', engine='python', header=None, names=['ID', 'Abstract']).dropna()
abstracts = abstracts[abstracts['ID'].astype(str).str.isnumeric()].astype({'ID': 'uint32'})
abstracts['Abstract'] = abstracts['Abstract'].fillna('')

edges = pd.read_csv(base_path + 'edgelist.txt', sep=',', header=None, names=['PaperID1', 'PaperID2']).dropna()
edges = edges[edges['PaperID1'].astype(str).str.isnumeric() & edges['PaperID2'].astype(str).str.isnumeric()].astype({'PaperID1': 'uint32', 'PaperID2': 'uint32'})

authors = pd.read_csv(base_path + 'authors.txt', sep=r'\|--\|', engine='python', header=None, names=['PaperID', 'AuthorNames']).dropna()
authors['AuthorNames'] = authors['AuthorNames'].astype(str).str.split(',')
authors = authors.explode('AuthorNames')
authors = authors[authors['PaperID'].astype(str).str.isnumeric()].astype({'PaperID': 'uint32'})

test = pd.read_csv(base_path + 'test.txt', sep=',', header=None, names=['PaperID1', 'PaperID2'])


model = SentenceTransformer('BAAI/bge-large-en')
embeddings_df = pd.read_csv(os.path.join(base_path, 'embeddings.csv'))
embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
abstracts = abstracts.merge(embeddings_df, on="ID", how="inner")

#Author Dictionary
author_dict = authors.groupby('PaperID')['AuthorNames'].agg(set).to_dict()

#Graph Features
G = nx.Graph()
G.add_edges_from(zip(edges['PaperID1'], edges['PaperID2']))
pagerank = nx.pagerank(G)
degree = nx.degree_centrality(G)
citation_set = set(zip(edges['PaperID1'], edges['PaperID2']))

#Similarity Calculation
def fuzzy_jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    scores = [max(fuzz.ratio(a1, a2) for a2 in set2) for a1 in set1]
    return np.mean(scores) / 100

results = []
for row in test.itertuples(index=False):
    p1, p2 = row.PaperID1, row.PaperID2
    vec1 = abstracts.loc[abstracts['ID'] == p1, 'Embedding'].values
    vec2 = abstracts.loc[abstracts['ID'] == p2, 'Embedding'].values

    if len(vec1) == 0 or len(vec2) == 0:
        sim = 0
    else:
        sim = 1 - cosine(vec1[0], vec2[0])
    sim = np.log1p(sim)

    a1, a2 = author_dict.get(p1, set()), author_dict.get(p2, set())
    sim += 0.2 * fuzzy_jaccard(a1, a2)

    if (p1, p2) in citation_set or (p2, p1) in citation_set:
        sim += 0.2
    sim += 0.1 * pagerank.get(p1, 0) + 0.1 * pagerank.get(p2, 0)
    sim += 0.05 * degree.get(p1, 0) + 0.05 * degree.get(p2, 0)

    results.append(min(1.0, max(0.0, sim)))

#Save Submission
test['Similarity'] = results
test['ID'] = np.arange(len(test), dtype='uint32')
test[['ID', 'Similarity']].to_csv("test_submission_sbert_gat.csv", index=False, encoding='utf-8')
