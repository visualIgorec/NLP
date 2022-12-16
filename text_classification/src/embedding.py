from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def embeddings(
    raw_data,
    max_f = 512,
    get_projection = False,
    ):

    ### form embeddings from text ###

    tfidf = TfidfVectorizer(stop_words=None, binary=True, max_features = max_f) 
    text_embeddings = tfidf.fit_transform(raw_data['text']).toarray()
    target = raw_data['topic'].to_numpy()

    if get_projection:
        # преобразуем в 2д для визуализации кластеров
        umap = UMAP()
        embed_2d = umap.fit_transform(text_embeddings)

        # кластеризация 
        kmeans = KMeans(n_clusters = raw_data['topic'].unique().__len__())
        kmeans.fit(embed_2d)
        raw_data['cluster'] = kmeans.labels_

        clu_name = dict(zip(raw_data['cluster'].unique(), raw_data['topic'].unique()))

        # Display Topics
        centers = kmeans.cluster_centers_
        plt.figure(figsize=(6,6))
        plt.scatter(embed_2d[:,0], embed_2d[:,1], s=1, c=kmeans.labels_)
        plt.title('Кластеризация по топикам', size=16)

        for k in range(len(centers)):
            plt.text(centers[k,0]-1, centers[k,1] + 0.75, f'{k+1}-{clu_name[k]}', size=12)
        plt.show()

    return text_embeddings, target





