import numpy as np

# class KNN:
#   def __init__(self, k=3):
#     self.k = k
#     self.X_train = None
#     self.y_train = None

#   def fit(self, X, y):
#     self.X_train = X
#     self.y_train = y

#   def predict(self, X):
#     predictions = [self._predict(x) for x in X]
#     return np.array(predictions)
  
#   def euclidean_distance(self, x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

#   def _predict(self, x):
#     # Step 1: Compute the distances of nearest neighbors of the query.
#     distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
#     # Step 2: Sort the distances in an ascending order.
#     k_indices = np.argsort(distances)[:self.k]
#     # Step 3: Search k-nearest neighbors of the query.
#     k_nearest_labels = [self.y_train[i] for i in k_indices]
#     # Step 4: Assign a majority weighted voting class label to the query.
#     unique_labels = np.unique(k_nearest_labels)
#     label_counts = [np.sum(np.equal(k_nearest_labels, label).astype(int)) for label in unique_labels]
#     best_label = unique_labels[np.argmax(label_counts)]
#     return best_label

class DWKNN:
  def __init__(self, k=3, metric='euclidean'):
    self.k = k
    self.metric = metric
    self.X_train = None
    self.y_train = None

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return np.array(predictions)
  
  def euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
  
  def cosine_similarity(self, x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

  def _predict(self, x):
    # Step 1: Compute the distances of nearest neighbors of the query.
    if self.metric == 'cosine_similarity':
      distances = [self.cosine_similarity(x, x_train) for x_train in self.X_train]
    else:
      distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
    # Step 2: Sort the distances in an ascending order.
    k_indices = np.argsort(distances)[:self.k]
    # Step 3: Search k-nearest neighbors of the query.
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    # Step 4: Calculate the dual weights of k nearest neighbors.
    k_distances = [distances[i] for i in k_indices]
    if k_distances[-1] != k_distances[0]:
      weights = [(k_distances[-1] - d) / (k_distances[-1] - k_distances[0]) * (k_distances[-1] + k_distances[0]) / (k_distances[-1] + d) for d in k_distances]
    else:
      weights = [1 for _ in range(self.k)]
    # Step 5: Assign a majority weighted voting class label to the query.
    unique_labels = np.unique(k_nearest_labels)
    weighted_counts = [np.sum(weights * np.equal(k_nearest_labels, label).astype(int)) for label in unique_labels]
    best_label = unique_labels[np.argmax(weighted_counts)]
    return best_label