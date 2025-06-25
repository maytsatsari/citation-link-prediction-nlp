#  Semantic Citation Matching with NLP and Graph Features

 **Project**: Citation Link Prediction – NLP Course Challenge  
 **Course**: Natural Language Processing   
 **Author**: Maria Tsatsari

This repository presents a complete solution to a **citation link prediction task**, where the goal is to determine whether one research paper cites another, based on their abstracts, titles, and metadata.

The project integrates **text-based semantic similarity**, **graph-based features**, and **ensemble logic**, using a wide range of NLP and ML techniques.

>  Only representative code is included. Several experimental variants are archived offline to maintain clarity.

---

##  Problem Overview

Given two papers (A, B), predict the probability that **paper A cites paper B**. This is modeled as a **binary classification** problem and framed as **link prediction** on a citation graph.

---

##  Kaggle-style Competition Format

The challenge was structured as a **Kaggle-style competition** within the course.  
Participants submitted `.csv` files with probabilities for each paper pair. Evaluation was done using **ROC-AUC**, with a public/private leaderboard.


---

##  Techniques Used

### Features
- **SBERT / SPECTER** sentence/document embeddings
- **Word2Vec**, **TF-IDF** representations
- Cosine similarity, Jaccard similarity, abstract length, title overlap
- Author matching with fuzzy logic
- Graph features: PageRank, edge existence, citation direction

###  Models
- Logistic Regression, Random Forest, XGBoost
- Ensemble Stacking
- Calibrated Classifiers (Platt scaling)
- GAT (Graph Attention Network – PyTorch Geometric)

---

##  Repository Contents

```
semantic-citation-matching/
├── wdvec.py             # Word2Vec + feature-based prediction
├── wdvec2.py            # Extended version with Jaccard, author overlap
├── tf-wvec.py           # Hybrid TF-IDF + Word2Vec approach
├── sbert_specter.py     # Cosine similarity using SBERT and SPECTER
├── 0.50.py              # Rule-based model combining SBERT + author matching
├── 0.566.py             # Final submission script with full ensemble logic
├── Report.pdf         # Full report with experiments & results
└── README.md
```

---

##  Example Run

```bash
# Generate predictions with SBERT + graph + ensemble features
python 0.566.py --input data/ --output final_submission.csv
```

> Note: Some scripts require preprocessed embedding files and metadata (e.g., authors, edgelist, abstract vectors).

---

## Results

- Best model: Ensemble of SBERT + citation edges + author Jaccard + scaling


---


