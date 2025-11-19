import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class K_means:
    def __init__(self, n_clusters=3, max_iters=1000, tol=1e-4, patience=10):
        self.patience = patience
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = 42
        self.centroids = None
        self.labels = None
        self.l = 0
    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0] 
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices]
    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        centroids = []
        for i in range(self.n_clusters):
            if np.any(labels == i):
                centroid = X[labels == i].mean(axis=0)
            else:
                centroid = np.zeros(X.shape[1])
            centroids.append(centroid)
        return np.array(centroids)
    def compute_l(self, X, labels, centroids):
        return sum(np.linalg.norm(X[i] - centroids[labels[i]]) for i in range(len(X)))   

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        patience_counter = 0

        for _ in range(self.max_iters):
            labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, labels)

            # verifico si los centroides no cambiaron bajo cierta tolerancia
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            else:
                patience_counter = 0

            self.centroids = new_centroids

        self.labels = self.assign_clusters(X, self.centroids)
        self.l = self.compute_l(X, self.labels, self.centroids)
        return self.centroids, self.labels, self.l


    def predict(self, X):
        return self.assign_clusters(X, self.centroids)
    

class GMM:
    def __init__(self, k, tol=1e-4, max_iter=1000, patience=5):
        self.patience = patience
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        model = K_means(self.k)
        centroids, _, _ = model.fit(X)
        self.means = centroids # inicialización con KMeans
        self.covs = [np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.k)]
        self.weights = np.ones(self.k) / self.k

    def e_step(self, X):
        n_samples = X.shape[0]
        self.resp = np.zeros((n_samples, self.k))

        for i in range(self.k):
            rv = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            self.resp[:, i] = self.weights[i] * rv.pdf(X)

        self.resp /= np.sum(self.resp, axis=1, keepdims=True)

    def m_step(self, X):
        n_samples, n_features = X.shape

        for i in range(self.k):
            r_i = self.resp[:, i].reshape(-1, 1)
            total_r = np.sum(r_i)
            self.means[i] = np.sum(r_i * X, axis=0) / total_r
            diff = X - self.means[i]
            self.covs[i] = (r_i * diff).T @ diff / total_r + 1e-6 * np.eye(n_features)
            self.weights[i] = total_r / n_samples

    def compute_L(self, X):
        closest_centroids = np.argmax(self.resp, axis=1)
        L = 0
        for i in range(self.k):
            cluster_points = X[closest_centroids == i]
            if len(cluster_points) > 0:
                dists = np.linalg.norm(cluster_points - self.means[i], axis=1)
                L += np.sum(dists)
        return L
    def fit(self, X):
        self.initialize_parameters(X)
        patience_counter = 0

        prev_means = self.means.copy()
        prev_covs = [cov.copy() for cov in self.covs]
        prev_weights = self.weights.copy()

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            L = self.compute_L(X)

            if not self.parameters_changed(prev_means, prev_covs, prev_weights, self.tol):
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Converged (parameters stable) at iteration {iteration}")
                    break
            else:
                patience_counter = 0

            prev_means = self.means.copy()
            prev_covs = [cov.copy() for cov in self.covs]
            prev_weights = self.weights.copy()

        return L

    def parameters_changed(self, prev_means, prev_covs, prev_weights, tol):
        if np.linalg.norm(self.means - prev_means) > tol:
            return True
        if np.linalg.norm(self.weights - prev_weights) > tol:
            return True
        for cov, prev_cov in zip(self.covs, prev_covs):
            if np.linalg.norm(cov - prev_cov) > tol:
                return True
        return False
    def predict(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.k))

        for i in range(self.k):
            rv = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            probs[:, i] = self.weights[i] * rv.pdf(X)

        return np.argmax(probs, axis=1)
class DBSCAN:
    def __init__(self, r=0.5, min_pts=5):
        self.r = r
        self.min_pts = min_pts
        self.labels = None

    def fit(self, X):
        n = X.shape[0]
        labels = np.full(n, -1)  # -1: ruido
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0


        dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = np.where(dists[i] <= self.r)[0]

            if len(neighbors) < self.min_pts:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                queue = list(neighbors)
                while queue:
                    j = queue.pop(0)
                    if not visited[j]:
                        visited[j] = True
                        neighbors_j = np.where(dists[j] <= self.r)[0]
                        if len(neighbors_j) >= self.min_pts:
                            queue.extend(neighbors_j.tolist())
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1

        self.labels = labels

    def get_labels(self):
        return self.labels

    def silhouette_loss(self, X):
        labels = self.labels
        unique_labels = np.unique(labels[labels != -1])
        n_clusters = len(unique_labels)
        if n_clusters < 2:
            return 0.0

        dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        silhouette_scores = []

        for i in range(X.shape[0]):
            label = labels[i]
            if label == -1:
                continue 

            same_cluster = (labels == label)
            a = np.mean(dists[i][same_cluster & (np.arange(len(labels)) != i)])

            b = np.inf
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_cluster = (labels == other_label)
                b_candidate = np.mean(dists[i][other_cluster])
                if b_candidate < b:
                    b = b_candidate

            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_scores.append(s)

        return -np.mean(silhouette_scores)  # negativo para usar como "loss"
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None

    def fit(self, X):
        cov_matrix = np.cov(X, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eig_vals)[::-1]
        self.eigenvalues = eig_vals[sorted_idx]
        self.components = eig_vecs[:, sorted_idx[:self.n_components]]
        return self

    def transform(self, X):
        return X @ self.components

    def inverse_transform(self, X_proj):
        X_rec = X_proj @ self.components.T
        return X_rec

    def reconstruction_error(self, X):
        X_proj = self.transform(X)
        X_rec = self.inverse_transform(X_proj)
        return np.mean((X - X_rec) ** 2)
    def best_components(self):
        # Calcular varianza acumulada de los aval
        eigenvalues = self.eigenvalues
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Encontrar el menor número de componentes que expliquen al menos el 80% (sacado de clase de martin)
        target_variance = 0.80
        n_components_80 = np.argmax(cumulative_variance >= target_variance) + 1
        return n_components_80

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28 * 28)))
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.fc2 = nn.Linear(400, 28 * 28)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(z)).view(-1, 1, 28, 28)

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

