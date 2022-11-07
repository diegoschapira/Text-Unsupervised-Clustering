import pandas as pd
import numpy as np

#conda install -c conda-forge sentence-transformers
import torch
from sentence_transformers import SentenceTransformer, util

#!pip install umap-learn
import umap
import umap.umap_ as umap

# Clustering
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# NLP Packages
import nltk
from nltk.tokenize import sent_tokenize

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Load BERT Multilingual via Sentence BERT >> SIAMESE NETWORK using BERT.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2').to(device)

model_sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

path = 'data.xlsx'
df = pd.read_excel(path)
df.head()

#Data Prep
text_list = df['text'].tolist()
sentences = [sent_tokenize(i) for i in text_list]
sentences_list = [item for sublist in sentences for item in sublist]

#Generate embeddings for Sentences
batch_size=128
embeddings_sbert_t = model_sbert.encode(sentences_list, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True) 
embeddings_sbert_t = torch.stack([t.cpu() for t in embeddings_sbert_t])
embeddings_ = [t.detach().numpy() for t in embeddings_sbert_t] #s_bert

# Dimensionality Reduction (UMAP) + Clustering (HDBSCAN)
umap_embeddings = umap.UMAP(min_dist=0, metric='cosine', random_state=42).fit_transform(embeddings_) 

final_cluster = hdbscan.HDBSCAN(min_cluster_size = 15, 
                                           min_samples = 5,
                                           metric='euclidean', 
                                           gen_min_span_tree=True,
                                           cluster_selection_method="eom").fit(umap_embeddings)

#Transforming to df for unstacking and join
df_sentences = pd.DataFrame({"Sentences" : sentences})

#Unstacking...
df__sentences = pd.DataFrame({'Index':np.repeat(df_sentences.index.values, df_sentences.Sentences.str.len()),
              'Sentences':[x for sublist in sentences for x in sublist]})

df__sentences.set_index('Index', inplace = True)
df__sentences["CLUSTER"] = final_cluster.labels_
df__sentences["PROBABILITY"] = final_cluster.probabilities_
df__sentences['CLUSTER'] = df__sentences['CLUSTER'].replace(-1,999)
df__sentences["UMAP_X"] = umap_embeddings[:,0]
df__sentences["UMAP_Y"] = umap_embeddings[:,1]
df__sentences.head()


# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_facecolor('white')


outliers = df__sentences.loc[df__sentences.CLUSTER == 999, :]
clustered = df__sentences.loc[df__sentences.CLUSTER != 999, :]
plt.scatter(outliers.UMAP_X, outliers.UMAP_Y, color='#BDBDBD', s=0.1)
plt.scatter(clustered.UMAP_X, clustered.UMAP_Y, c=clustered.CLUSTER, s=0.5, cmap='hsv_r')
plt.grid(False)
plt.colorbar()
