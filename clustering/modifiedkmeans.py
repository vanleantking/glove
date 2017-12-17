from bikmeans import KMeans
from dotdict import dotdict
import numpy as np

class KMeansModified(KMeans):

    def execute(self, X):
        clusters = []
        cluster = dotdict()
        cluster.vectors = []
        for x in X:
            cluster.vectors.append(x)
        
        cluster.centroid = self.calculate_centroid(cluster.vectors)
       
        clusters.append(cluster)

        o_centroids = [cluster['centroid'] for cluster in clusters]
        n_centroids = o_centroids
        count = 1        
            
        while True:
            final_clusters = []
            while True:
                o_centroids = n_centroids
                clusters = self.assign_point_to_clusters(X, n_centroids)
                n_centroids = self.recalculate_centroids(clusters)
                if (self.has_converged(n_centroids, o_centroids)):
                    for index in range(len(n_centroids)):
                        if index in clusters.keys():
                            cluster = dotdict()
                            cluster.centroid = n_centroids[index]
                            cluster.vectors = clusters[index]
                            final_clusters.append(cluster)
                    break

            max_intra_cluter, max_centroid, max_cluster = self.get_max_intra_cluter(final_clusters)
            if max_intra_cluter < 0.2:
                break

            max_distance = self.get_max_distance(max_cluster)
            for index, centroid in enumerate(n_centroids):
                if np.array_equal(centroid, max_centroid):
                    del n_centroids[index] 
            n_centroids.extend(max_distance)

        
        return final_clusters


    def recalculate_centroids(self, clusters):

        centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            if len(clusters[k]) > 0:
                centroids.append(np.mean(clusters[k], axis = 0))
        
        return centroids


    def has_converged(self, new_centers, old_centers):
        return (set([tuple(a) for a in old_centers if a is not None]) == set([tuple(a) for a in new_centers if a is not None]))


    def bi_convegence(self, clusters, old_clusters):
        clusters = { k : v for k,v in clusters.items() if v}
        return (set([tuple(v) for k, v in clusters.items()]) == set([tuple(a['vectors']) for a in old_clusters]))

    def calculate_centroid(self, clusters):
        return list(np.mean(clusters, axis=0))


    def get_max_intra_cluter(self, clusters):
        max_intra_cluters = []
        for cluster in clusters:
            sim = 0.0
            centroid = cluster['centroid']
            if len(cluster['vectors']) > 0:
                for vector in cluster['vectors']:
                    sim += self.cosine_similitary(vector, centroid)
                max_intra_cluters.append(sim/len(cluster['vectors']))
            else:
                max_intra_cluters.append(float("-inf"))

        max_intra = float("-inf")
        max_centroid = None
        max_cluster = None
        for index, intra_cluster in enumerate(max_intra_cluters):
            if intra_cluster > max_intra:
                max_intra = intra_cluster
                max_cluster = clusters[index]['vectors']
                max_centroid = clusters[index]['centroid']
        
        return max_intra, max_centroid, max_cluster

    def get_max_distance(self, cluster):
        distances = 0
        max_distance =  float("-inf")
        vectors_max = []
        for v_i in range(0, len(cluster) - 1):
            for v_j in range(v_i + 1, len(cluster)):
                distance = self.cosine_similitary(cluster[v_i], cluster[v_j])
                if distance > max_distance:
                    max_distance = distance
                    vectors_max.extend([cluster[v_i], cluster[v_j]])
        return vectors_max

            