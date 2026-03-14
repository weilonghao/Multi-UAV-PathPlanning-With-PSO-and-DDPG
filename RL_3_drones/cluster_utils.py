#cluster_utils.py
import numpy as np
from scipy.spatial.distance import euclidean
np.random.seed(42)
def perform_clustering(targets_dict, R=500, original_survival=None):

    if original_survival is None:
        n = len(targets_dict)
        original_survival = [1.0] * n

    targets = list(targets_dict.values())
    n = len(targets)

    # 计算距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean(targets[i], targets[j])

    # 构建邻域关系
    neighborhoods = {
        i: [j for j in range(n) if j != i and dist_matrix[i][j] < R]
        for i in range(n)
    }

    # 迭代聚类
    clusters = []
    assigned = set()
    while len(assigned) < n:
        candidates = [
            (i, len([j for j in neighborhoods[i] if j not in assigned]))
            for i in range(n) if i not in assigned
        ]
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        cluster = [best_candidate] + [j for j in neighborhoods[best_candidate] if j not in assigned]
        clusters.append(cluster)
        assigned.update(cluster)

    # 计算聚类中心
    cluster_centers = []
    for cluster in clusters:
        x = np.mean([targets[i][0] for i in cluster])
        y = np.mean([targets[i][1] for i in cluster])
        cluster_centers.append((round(x, 2), round(y, 2)))

    # 计算聚类后的生存概率
    clustered_survival = [
        np.mean([original_survival[i] for i in cluster])
        for cluster in clusters
    ]

    return cluster_centers, clusters, clustered_survival
