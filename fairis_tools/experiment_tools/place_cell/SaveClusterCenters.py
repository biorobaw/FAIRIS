import pickle
import os
import numpy as np
from sklearn.cluster import KMeans, Birch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
os.chdir("../../..")
print(os.getcwd())

with open("data/VisualPlaceCellData/LM8_1000",'rb') as file:
    visual_place_cell_data = pickle.load(file)


n_clusters = 18  # You can change this based on how many clusters you expect

# Function to plot clusters on the x, y plane
def plot_clusters(xy_list, cluster_labels, name):
    """
    Plot the clusters on the x, y plane using the original (x, y) coordinates, and save the figure.

    Args:
    - xy_list (list of tuples): The original x, y coordinates for each datapoint.
    - cluster_labels (list of int): The cluster label for each datapoint.
    - name (str): The filename to save the figure as (e.g., "clusters_plot.png").
    """
    # Convert xy_list to NumPy arrays for easy plotting
    x_coords = np.array([x for x, y in xy_list])
    y_coords = np.array([y for x, y in xy_list])

    # Check if the lengths match
    if len(x_coords) != len(cluster_labels):
        raise ValueError(f"Mismatch: {len(x_coords)} coordinates and {len(cluster_labels)} cluster labels.")

    # Scatter plot with color coding for clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, c=cluster_labels, cmap='rainbow', alpha=0.7)

    # Add color bar to indicate clusters
    plt.colorbar(scatter, label='Cluster')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Clustering with {len(set(cluster_labels))} Clusters on X-Y Plane')

    # Save the plot to the specified file
    plt.savefig(name)

    # Close the plot to avoid displaying it when running in scripts
    plt.close()


def plot_clusters_by_subplots(xy_list, cluster_labels, name, n_clusters=8):
    """
    Plot the clusters on 8 subplots, one for each cluster, using the original (x, y) coordinates.

    Args:
    - xy_list (list of tuples): The original x, y coordinates for each datapoint.
    - cluster_labels (list of int): The cluster label for each datapoint.
    - n_clusters (int): Number of clusters to plot (default is 8).
    """
    # Convert xy_list to NumPy arrays for easy filtering and plotting
    x_coords = np.array([x for x, y in xy_list])
    y_coords = np.array([y for x, y in xy_list])

    # Create subplots (arranged as 4 rows, 2 columns for 8 clusters)
    fig, axes = plt.subplots(3, 6, figsize=(30, 20))  # 4 rows, 2 columns for 8 clusters
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each cluster on its respective subplot
    for cluster_id in range(n_clusters):
        # Get the indices of the datapoints that belong to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Filter x and y coordinates for the current cluster
        cluster_x = x_coords[cluster_indices]
        cluster_y = y_coords[cluster_indices]

        # Scatter plot for the current cluster
        axes[cluster_id].scatter(cluster_x, cluster_y, c=f'C{cluster_id}', alpha=0.7)

        # Set subplot title and labels
        axes[cluster_id].set_title(f'Cluster {cluster_id}')
        axes[cluster_id].set_xlabel('X Coordinate')
        axes[cluster_id].set_ylabel('Y Coordinate')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    fig.savefig(name)
    plt.close()

def format_data_for_clustering(data):
    multimodal_feature_vectors = []
    cnn_feature_vectors = []
    xy_list = []
    theta_list = []
    for observation in data.observations:
        multimodal_feature_vectors.append(observation.multimodal_feature_vector)
        cnn_feature_vectors.append(observation.cnn_feature_vector)
        xy_list.append((observation.x, observation.y))
        theta_list.append(observation.theta)

    return multimodal_feature_vectors, cnn_feature_vectors, xy_list, theta_list

multimodal_feature_vectors,cnn_feature_vectors,xy_list,theta_list = format_data_for_clustering(visual_place_cell_data)
def cluster_with_kmeans_and_save_centers(features_list, n_clusters, centers_save_path):
    """
    Perform KMeans clustering, calculate the maximum distance for each cluster,
    and save the cluster centers and max distances as a list of lists using pickle.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to form.
    - centers_save_path (str): Path to save the cluster centers and max distances using pickle.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    - cluster_centers (numpy array): The centers of the final clusters.
    """
    features_array = np.array(features_list)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_array)

    # Get cluster centers and labels for each data point
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # List to store [center, max_distance] for each cluster
    cluster_data = []

    # Calculate max distance for each cluster
    for cluster_index in range(n_clusters):
        # Get data points belonging to this cluster
        cluster_points = features_array[labels == cluster_index]

        # Calculate distances from each point to the cluster center
        distances = cdist(cluster_points, [cluster_centers[cluster_index]], metric='euclidean').flatten()

        # Find the maximum distance for this cluster
        max_distance = distances.max()

        # Append the center and max distance as a pair to cluster_data
        cluster_data.append([cluster_centers[cluster_index].tolist(), max_distance])

    # Save cluster_data (centers and max distances) using pickle
    with open(centers_save_path, 'wb') as f:
        pickle.dump(cluster_data, f)

    return cluster_labels

centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/multimodal_kmeans_centers"
cluster_labels = cluster_with_kmeans_and_save_centers(multimodal_feature_vectors, n_clusters, centers_save_path)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/multi_model_knn.png")
plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/multimodel_knn_clusters.png", n_clusters=n_clusters)

centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/cnn_kmeans_centers"
cluster_labels = cluster_with_kmeans_and_save_centers(cnn_feature_vectors, n_clusters, centers_save_path)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/cnn_knn.png")
plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/cnn_knn_clusters.png", n_clusters=n_clusters)

# Function to perform Birch clustering and save the model and cluster centers
def cluster_with_birch_and_save_centers(features_list, n_clusters, model_save_path, centers_save_path):
    """
    Perform Birch clustering, approximate the cluster centers by averaging the points in each cluster,
    calculate the maximum distance of points to their cluster center, and save the centers with distances.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to form.
    - model_save_path (str): Path to save the trained Birch model using pickle.
    - centers_save_path (str): Path to save the cluster centers and max distances as a Python list using pickle.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    - cluster_centers (numpy array): The approximate centers of the final clusters.
    """
    features_array = np.array(features_list)

    # Perform Birch clustering
    birch = Birch(n_clusters=n_clusters)
    cluster_labels = birch.fit_predict(features_array)

    # Save the Birch model
    with open(model_save_path, 'wb') as f:
        pickle.dump(birch, f)

    # List to store [center, max_distance] for each cluster
    cluster_data = []

    # For each final cluster, calculate the center and max distance
    for cluster_id in range(n_clusters):
        # Get the feature vectors for the points in this cluster
        cluster_points = features_array[cluster_labels == cluster_id]

        # Compute the center by averaging the points in this cluster
        if len(cluster_points) > 0:
            cluster_center = np.mean(cluster_points, axis=0)
        else:
            # In case a cluster has no points (unlikely), set a zero vector
            cluster_center = np.zeros(features_array.shape[1])

        # Calculate distances from each point to the cluster center
        distances = cdist(cluster_points, [cluster_center], metric='euclidean').flatten()

        # Find the maximum distance
        max_distance = distances.max() if len(distances) > 0 else 0

        # Append the center and max distance as a pair to cluster_data
        cluster_data.append([cluster_center.tolist(), max_distance])

    # Save cluster_data (centers and max distances) using pickle
    with open(centers_save_path, 'wb') as f:
        pickle.dump(cluster_data, f)

    return cluster_labels

centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/multimodal_birch_centers"
cluster_labels = cluster_with_kmeans_and_save_centers(multimodal_feature_vectors, n_clusters, centers_save_path)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/multi_model_birch.png")
plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/multimodel_birch_clusters.png", n_clusters=n_clusters)

centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/cnn_birch_centers"
cluster_labels = cluster_with_kmeans_and_save_centers(cnn_feature_vectors, n_clusters, centers_save_path)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/cnn_birch.png")
plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/cnn_birch_clusters.png", n_clusters=n_clusters)