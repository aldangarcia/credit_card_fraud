from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def pca_component_selection(original_data):
    # Step 1: Preprocess data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data)

    # Step 2: Apply PCA
    pca = PCA(n_components=None)  # Calculate all components
    principal_components = pca.fit_transform(scaled_data)

    # Step 3: Analyze variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Components')
    plt.show()

    # Step 4: Select components and transform data
    n_selected_components = np.argmax(cumulative_variance >= 0.95) + 1
    pca = PCA(n_components=n_selected_components)
    reduced_data = pca.fit_transform(scaled_data)

    print(f"Reduced to {n_selected_components} components.")