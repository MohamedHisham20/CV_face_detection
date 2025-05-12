import numpy as np


class Custom_PCA:
    def __init__(self, n_components=None, explained_variance_ratio=0.95) -> None:
        self.n_components = n_components
        self.explained_variance_ratio = explained_variance_ratio
        self.mean_face = None
        self.components_ = None

    def fit_transform(self, X) -> np.ndarray:
        if self.components_ is None:
            self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        return np.dot(X, self.components_.T) + self.mean_face

    def fit(self, X) -> None:
        self.mean_face = np.mean(X, axis=0)
        X_centered = X - self.mean_face

        # cov_matrix = np.cov(X_centered, rowvar=False)
        # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # eigenvectors = eigenvectors.T

        # without computing the full covariance matrix
        # use svd (Singular Value Decomposition) -> much much faster.

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues = S**2 / (X_centered.shape[0] - 1)
        eigenvectors = Vt.T

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if self.n_components is None:
            cumulative_variance = np.cumsum(eigenvalues / np.sum(eigenvalues))
            self.n_components = np.argmax(cumulative_variance >= self.explained_variance_ratio) + 1
            print(f"Using {self.n_components} components to explain {self.explained_variance_ratio * 100}% of the variance.")

        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        print(f"Corrected explained variance ratio: {self.explained_variance_ratio}")

    def transform(self, X) -> np.ndarray:
        X_centered = X - self.mean_face
        return np.dot(X_centered, self.components_)

    def load(self, filename):
        data = np.load(filename)
        self.mean_face = data['mean_face']
        self.explained_variance_ratio = data['explained_variance_ratio']
        self.components_ = data['components']
        self.n_components = self.components_.shape[0]
        print(f"Loaded {self.n_components} components from {filename}")

    def save(self, filename):
        np.savez(filename, mean_face=self.mean_face, explained_variance_ratio=self.explained_variance_ratio, components=self.components_)
        print(f"Saved {self.n_components} components to {filename}")
