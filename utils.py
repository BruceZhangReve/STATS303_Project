import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
def reject_sampling(A, num_samples):
    samples = []
    pca = PCA(n_components=20）
    pca.fit(A)
    image_pca = pca.transform(A)
    eigenvalues = pca.explained_variance_
    weights = eigenvalues / np.sum(eigenvalues)

    while len(samples) < num_samples:
        weights_noisy = weights + np.random.normal(0, 1 / (np.arange(20)**2+1))
        g = np.dot(image_pca, weights)
        gn = np.dot(image_pca, weights_noisy)

        if cosine_similarity([g], [gn])[0][0] >= 0.6:
            samples.append(gn)

    return np.array(samples)

def subseting_samples(data, num_samples_to_select):
    num_samples = data.shape[1]
    selected_indices = np.random.choice(num_samples, num_samples_to_select, replace=False)
    selected_samples = data[:,selected_indices]
    return selected_samples