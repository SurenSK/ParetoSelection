import numpy as np
from statistics import mean, variance
import time
from scipy import stats
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

def normalize(lst):
    lst = np.array(lst)
    min_val = np.min(lst)
    max_val = np.max(lst)
    return (lst - min_val) / (max_val - min_val)

def genPoints(n, d, dist='uniform'):
    if dist == 'uniform':
        data = np.random.uniform(0, 1, (n, d))
    elif dist == 'normal':
        data = np.random.normal(0.5, 0.1, (n, d))
    elif dist == 'exponential':
        data = np.minimum(np.random.exponential(scale=0.1, size=(n, d)), 1)
    else:
        raise ValueError("Unsupported distribution type. Supported types: 'uniform', 'normal', 'exponential'")
    return [list(row) for row in data]  # Convert numpy arrays to lists for each point

def getGroundTruth(points):
    map_fit_ind = defaultdict(list)
    for point in points:
        map_fit_ind[tuple(point.values)].append(point)
    
    fits = list(map_fit_ind.keys())
    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            point_i = map_fit_ind[fit_i][0]
            point_j = map_fit_ind[fit_j][0]
            if point_i.dominates(point_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif point_j.dominates(point_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = []
    while current_front:
        fronts.append([p for fit in current_front for p in map_fit_ind[fit]])
        next_front = []
        for fit_p in current_front:
            for fit_d in dominated_fits[fit_p]:
                dominating_fits[fit_d] -= 1
                if dominating_fits[fit_d] == 0:
                    next_front.append(fit_d)
        current_front = next_front
    
    points_idx = {}
    for i,front in enumerate(fronts):
        for p in front:
            points_idx[p]=i
    idx_points = [points_idx[p] for p in points]
    return normalize(idx_points)

def calcSVM(points):
    X = np.array([p.values for p in points])
    
    # Identify Pareto optimal points
    is_pareto = np.ones(X.shape[0], dtype=bool)
    for i, point in enumerate(X):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(X[is_pareto] > point, axis=1)
            is_pareto[i] = True
    
    # Use only Pareto optimal points for fitting
    X_pareto = X[is_pareto]
    
    # Fit the model with Pareto optimal points
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1).fit(X_pareto)
    
    # Calculate distances to the decision boundary for all points
    distances = model.decision_function(X)
    
    # Set distances for support vectors (Pareto optimal points) to 0
    support_vector_indices = model.support_
    for index in support_vector_indices:
        distances[index] = 0
    
    # Normalize the distances
    scaler = MinMaxScaler()
    normalized_distances = scaler.fit_transform(distances.reshape(-1, 1)).flatten().tolist()
    
    return normalized_distances

def scoreMethod(gt, points, method):
    t0 = time.time()
    predicted = method(points)
    tCalc = time.time()-t0
    errors = [abs(a-b) for a,b in zip(gt, predicted)]
    return sum(errors)/len(points), tCalc

# n, d = 100, 2
# points_array = genPoints(n, d)
# points = [Point(val) for val in points_array]
# err, t = scoreMethod(points, calcSVM)
class Point:
    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return str(self.values)

    def dominates(self, other):
        # Check if self dominates other
        return all(x >= y for x, y in zip(self.values, other.values)) and any(x > y for x, y in zip(self.values, other.values))

n = 1000  # Number of points
d = 5   # Dimension of each point
dist = 'uniform'  # Distribution type
points = [Point(val) for val in genPoints(n, d, dist)]

t0=time.time()
groundTruth = getGroundTruth(points)
tGT=time.time()-t0
err,t=scoreMethod(groundTruth, points, calcSVM)
print(err, t, tGT, tGT/t)