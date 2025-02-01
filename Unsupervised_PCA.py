import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    numerical_columns = ['Population denisty (sq km)', 'Average Elevation (m)',
                        'Average annual precipitiation (mm)', 'Mean temp in C',
                        'Extent of Forest (1000 ha)', 'Number of species',
                        'area (sq km)', 'latitude']

    X = df[numerical_columns].copy()

    for col in numerical_columns:
        if X[col].dtype == object:
            X.loc[:, col] = X[col].str.replace(',', '', regex=True).astype(float)
        else:
            X.loc[:, col] = X[col].astype(float)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df, numerical_columns

def pca_from_scratch(X, n_components=2):
    # Calculate covariance matrix
    cov_matrix = np.cov(X.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components]

    X_pca = np.dot(X, components)
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_pca, components, explained_variance_ratio, eigenvalues

def plot_scatter(X_pca, df, explained_variance_ratio):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Number of species'],
                         cmap='viridis', s=100)
    plt.xlabel(f'First PC ({explained_variance_ratio[0]:.2%} variance explained)')
    plt.ylabel(f'Second PC ({explained_variance_ratio[1]:.2%} variance explained)')
    plt.title('PCA Results with Number of Species Colored')
    plt.colorbar(scatter, label='Number of Species')

    #country labels
    for i, txt in enumerate(df['Country']):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

    plt.show()

def plot_scree(eigenvalues):
    plt.figure(figsize=(10, 6))
    variance_explained = (eigenvalues / np.sum(eigenvalues)) * 100
    print(f'principal component 3 = {variance_explained[2]}')
    plt.plot(range(1, len(eigenvalues) + 1), variance_explained, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.title('Scree Plot')
    plt.show()

def plot_loading_scores(components, numerical_columns):
    plt.figure(figsize=(10, 6))
    loading_scores = pd.DataFrame(
        components,
        columns=['PC1', 'PC2'],
        index=numerical_columns
    )
    sns.heatmap(loading_scores, annot=True, cmap='RdBu', center=0)
    plt.title('PCA Loading Scores')
    plt.show()

def plot_cumulative_variance(eigenvalues):
    plt.figure(figsize=(10, 6))
    variance_explained = (eigenvalues / np.sum(eigenvalues)) * 100
    cumulative_variance = np.cumsum(variance_explained)
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.title('Cumulative Variance Explained')
    plt.axhline(y=90, color='k', linestyle='--', alpha=0.5)
    plt.show()

###Building k-means algorithm by Bora

def k_means_algorithm():
    """Performs k-means clustering on PCA projections."""
    # Load and preprocess data
    X_scaled, df, numerical_columns = load_and_preprocess_data('Butterfly_Data_adjusted.csv')
    X_pca, components, explained_variance_ratio, eigenvalues = pca_from_scratch(X_scaled)
    PC1_proj = X_pca[:,0]
    PC2_proj = X_pca[:,1]
    PC_proj_stacked = np.column_stack((PC1_proj, PC2_proj))

    # Initialize clusters
    n_clusters = 3
    total_example_number = len(PC_proj_stacked)  # Dynamically determine total examples
    
    np.random.seed(42)
    rand_pick = np.random.choice(total_example_number, n_clusters, replace=False)
    centroids = PC_proj_stacked[rand_pick, :]

    # Iterative optimization of k-means
    for iteration in range(200):
        distance = np.linalg.norm(PC_proj_stacked[:, None, :] - centroids[None, :, :], axis=2)
        cluster_pick = np.argmin(distance, axis=1)

        new_centroids = np.array([
            PC_proj_stacked[cluster_pick == k].mean(axis=0) if np.any(cluster_pick == k) else centroids[k]
            for k in range(n_clusters)
        ])

        if np.allclose(centroids, new_centroids, atol=1e-4):
            print(f"{iteration+1} iterations required for convergence")
            break
        centroids = new_centroids

    return centroids, cluster_pick

def plot_k_means():
    """Plots the k-means clustering results on PCA projections."""
    # Redefine data
    X_scaled, df, numerical_columns = load_and_preprocess_data('Butterfly_Data_adjusted.csv')
    X_pca, components, explained_variance_ratio, eigenvalues = pca_from_scratch(X_scaled)
    PC1_proj = X_pca[:,0]
    PC2_proj = X_pca[:,1]
    PC_proj_stacked = np.column_stack((PC1_proj, PC2_proj))
    centroids, cluster_pick = k_means_algorithm()

    # Plotting
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for k in range(3):
        mask = cluster_pick == k  # Boolean mask for selecting points
        plt.scatter(PC_proj_stacked[mask, 0], PC_proj_stacked[mask, 1], 
                    label=labels[k], color=colors[k], alpha=0.7)

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Graph for K-means clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

##Final

def main():
    # Load and preprocess data
    X_scaled, df, numerical_columns = load_and_preprocess_data('Butterfly_Data_adjusted.csv')

    # Perform PCA
    X_pca, components, explained_variance_ratio, eigenvalues = pca_from_scratch(X_scaled)

    # Create and display visualizations
    plot_scatter(X_pca, df, explained_variance_ratio)
    plot_scree(eigenvalues)
    plot_loading_scores(components, numerical_columns)
    plot_cumulative_variance(eigenvalues)
    plot_k_means()

    # Print explained variance ratio
    print("\nExplained Variance Ratio:")
    print(f"PC1: {explained_variance_ratio[0]:.2%}")
    print(f"PC2: {explained_variance_ratio[1]:.2%}")
    print(f"PC3: {explained_variance_ratio[2]:.2%}")



if __name__ == "__main__":
    main()
        
        
          
        






    


