import pickle
import os
import numpy as np
from sklearn.cluster import KMeans, Birch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math
from fairis_tools.experiment_tools.loggers.visual_data_set import NavigationDataPoint
os.chdir("../../..")
print(os.getcwd())

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


def plot_clusters_by_subplots(xy_list, theta_list, cluster_labels, name, n_clusters=8):
    """
    Plot the clusters on subplots, one for each cluster, using the original (x, y) coordinates
    and their corresponding direction vectors.

    Args:
    - xy_list (list of tuples): The original x, y coordinates for each datapoint.
    - theta_list (list of floats): The orientation (theta) in degrees for each datapoint.
    - cluster_labels (list of int): The cluster label for each datapoint.
    - name (str): The filename to save the plot.
    - n_clusters (int): Number of clusters to plot.
    """
    # Fixed number of columns
    cols = 5
    # Calculate the number of rows needed
    rows = int(math.ceil(n_clusters / cols))

    # Dynamically scale the figsize based on rows and columns
    width_per_col = 6  # Adjust this for horizontal scaling
    height_per_row = 6  # Adjust this for vertical scaling
    figsize = (cols * width_per_col, rows * height_per_row)

    # Convert xy_list to NumPy arrays for easy filtering and plotting
    x_coords = np.array([x for x, y in xy_list])
    y_coords = np.array([y for x, y in xy_list])
    theta_list = np.array([theta for theta in theta_list])
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for cluster_id in range(n_clusters):
        # Get the indices of the datapoints that belong to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Filter x and y coordinates for the current cluster
        cluster_x = x_coords[cluster_indices]
        cluster_y = y_coords[cluster_indices]

        # Extract theta values for the current cluster
        cluster_theta = theta_list[cluster_indices]

        # Compute dx and dy for each point in the current cluster
        cluster_dx = 0.5 * np.cos(np.radians(cluster_theta))
        cluster_dy = 0.5 * np.sin(np.radians(cluster_theta))

        # Scatter plot for the current cluster
        axes[cluster_id].scatter(cluster_x, cluster_y, c=f'C{cluster_id}', alpha=0.7, label='Points')

        # Add quiver plot for vectors
        axes[cluster_id].quiver(cluster_x, cluster_y, cluster_dx, cluster_dy, angles='xy', scale_units='xy', scale=1,
                                color='black', alpha=0.7, label='Vectors')

        # Set subplot title and labels
        axes[cluster_id].set_title(f'Cluster {cluster_id}')
        axes[cluster_id].set_xlabel('X Coordinate')
        axes[cluster_id].set_ylabel('Y Coordinate')
        axes[cluster_id].set_xlim(-3, 3)  # Set x-axis limits
        axes[cluster_id].set_ylim(-3, 3)  # Set y-axis limits
        axes[cluster_id].legend()

    # Hide unused subplots (if n_clusters < len(axes))
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot to the specified file
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

with open("data/VisualPlaceCellData/LM8_Training",'rb') as file:
    visual_place_cell_data = pickle.load(file,encoding='latin1')

multimodal_feature_vectors,cnn_feature_vectors,xy_list,theta_list = format_data_for_clustering(visual_place_cell_data)
# n_clusters = [50,100,250,500]  # You can change this based on how many clusters you expect
n_clusters = [100]
for n_cluster in n_clusters:

    centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/multimodal_kmeans_"+str(n_cluster)+"centers_walls"
    cluster_labels = cluster_with_kmeans_and_save_centers(multimodal_feature_vectors, n_cluster, centers_save_path)

    # Now plot the clusters
    plot_clusters_by_subplots(xy_list, theta_list, cluster_labels, "data/figures/Clustering/multimodel_knn_"+str(n_cluster)+"clusters_walls.png", n_clusters=n_cluster)

    # centers_save_path = "data/VisualPlaceCellData/VisualPlaceCellClusters/cnn_kmeans_"+str(n_cluster)+"centers"
    # cluster_labels = cluster_with_kmeans_and_save_centers(cnn_feature_vectors, n_cluster, centers_save_path)
    #
    # # Now plot the clusters
    # plot_clusters_by_subplots(xy_list, theta_list, cluster_labels, "data/figures/Clustering/cnn_knn_"+str(n_cluster)+"clusters.png", n_clusters=n_cluster)
