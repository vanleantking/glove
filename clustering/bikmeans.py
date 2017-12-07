import math
import numpy as np
from scipy.spatial.distance import cdist
import random
from dotdict import dotdict

#from sklearn.cluster import KMeans
try:
  from itertools import izip as zip
except ImportError: # will be 3.x series
  pass

class KMeans:
    def __init__(self, k = 2, delta=.001, maxiter=300, metric='cosine'):
        self.k = k
        self.delta = delta
        self.maxiter = maxiter
        self.metric = metric

    def init_random_cluster(self, X):
    	#random choice k centroids
        return X[np.random.choice(X.shape[0], self.k, replace=False)]

    def has_converged(self, iterations, new_centers, old_centers):
        if self.maxiter == iterations:
           return True
        return (set([tuple(a) for a in old_centers if a is not None]) == set([tuple(a) for a in new_centers if a is not None]))


    def cosine_similitary(self, vector1, vector2):
        prod = 0.0
        
        mag1 = 0.0
        mag2 = 0.0
        
        for index, value in enumerate(vector1):
            prod += vector1[index] * vector2[index]
            mag1 += vector1[index] * vector1[index]
            mag2 += vector2[index] * vector2[index]
        
        return 1 - (prod / (math.sqrt(mag1) * math.sqrt(mag2)))

    def assign_point_to_clusters(self, X, centroids):
        clusters = {i : [] for i in range(self.k)}
        for x in X:
            mean_index = min([(m[0], self.cosine_similitary(x, centroids[m[0]])) for m in enumerate(centroids)], key=lambda t: t[1])[0]

            try: 
                clusters[mean_index].append(x)
            except KeyError:
                clusters[mean_index] = [x]

        return clusters
    
    def recalculate_centroids(self, clusters):

        centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            if len(clusters[k]) > 0:
                centroids.append(np.mean(clusters[k], axis = 0))
        
        return centroids
    
    def kmeans(self, X, k):
        old_centroids = X[np.random.choice(X.shape[0], size=k, replace=False), :]
        centroids = old_centroids
        iterations = 0
        lc_clusters = []
        while True:            
            old_centroids = centroids
            lc_clusters = self.assign_point_to_clusters(X = X, centroids = centroids)
            centroids = self.recalculate_centroids(lc_clusters)
            iterations +=1
            if (self.has_converged(iterations, centroids, old_centroids)):
                break

        final_clusters = []
        
        for index in range(len(centroids)):
            cluster = dotdict()
            cluster.centroid = centroids[index]
            cluster.vectors = lc_clusters[index]
            final_clusters.append(cluster)

        return final_clusters
        
    def similitary(self, clusters):
        sim = 0.0
        for cluster in clusters:
            centroid = cluster['centroid']
            for index in range(len(centroid)):
                sim += centroid[index]**2
        
        return sim / len(clusters)

class BiKMeans(KMeans):

    def __init__(self, k=2, max_iter = 500):
        self.k = k
        self.max_iter = max_iter
        self.clusters = []

    def dot_product(self,v1, v2):
        return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

    def cosine_measure(self, v1, v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return prod / (len1 * len2)

    def execute(self, X):
        clusters = []
        cluster = dotdict()
        results = []
        cluster.vectors = []
        for x in X:
            cluster.vectors.append(x)
        
        cluster.centroid = self.calculate_centroid(cluster.vectors)
       
        clusters.append(cluster)
        while True:
            split_cluster = self.find_smallest_sim_cluster(clusters)

            old_clusters = clusters
            # re-construct clusters except the split cluster
            clusters = [d for d in clusters if not np.array_equal(d['centroid'], split_cluster['centroid'])]
            max_cluster = float("-inf")
            max_bicluster = None
            for i in range(self.max_iter):
                kmeans = KMeans()

                # loop max_iter to find the best way to split
                biclusters = kmeans.kmeans(np.array(split_cluster.vectors), k = 2)
                sim = kmeans.similitary(biclusters)
                if (sim > max_cluster):
                    max_bicluster = [d for d in biclusters]
                    max_cluster = sim
            for index, clust in enumerate(biclusters):
                if len(clust['vectors']) == 1:
                    del biclusters[index]
                    results.append(clust)
            clusters.extend(biclusters)
            if (self.bi_convegence(clusters, old_clusters)):
                clusters.extend(results)
                break
            
        return clusters


    def convert_dotdict(self, datas):
        cluster = dotdict()
        cluster.vectors = []
        cluster.vectors[0] = datas[0]
        cluster.vectors[1] = datas[1]
        cluster.centroids = self.calculate_centroid(cluster.vectors)
        return datas


    def bi_convegence(self, clusters, old_clusters):
        for cluster in clusters:
            if (len(cluster['vectors']) == 1):
                return True
        return (set([tuple(a['centroid']) for a in clusters]) == set([tuple(a['centroid']) for a in old_clusters]))

    
    def calculate_centroid(self, clusters):
        return list(np.mean(clusters, axis=0))

    def find_smallest_sim_cluster(self, clusters):
        min_sim = float("inf")
        min_cluster = None
        for cluster in clusters:
            centroid = cluster['centroid']
            sim = 0.0
            for index in range(len(centroid)):
                sim += centroid[index]**2
            if sim < min_sim:
                min_sim = sim
                if len(cluster['vectors']) > 1:
                    min_cluster = cluster
        return min_cluster