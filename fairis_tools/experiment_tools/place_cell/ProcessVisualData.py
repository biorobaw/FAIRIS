import pickle
import torch
import torchvision.models as models
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
import matplotlib.pyplot as plt
os.chdir("../../..")
print(os.getcwd())

with open("data/VisualPlaceCellData/LM8_1000",'rb') as file:
    data = pickle.load(file)


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


# Load pre-trained ResNet50 model and remove classification layers for feature extraction
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])  # Remove the fully connected layers
resnet50.eval()  # Set model to evaluation mode


# Function to process a single image and extract features, with additional landmark and orientation data
def process_datapoint(image_tensor, landmarks_binary_list, orientation_radians):
    """
    Process a single image tensor through ResNet50, and augment the features with
    the binary list of detected landmarks and the robot's orientation.

    Args:
    - image_tensor (torch.Tensor): The image tensor from Webots camera.
    - landmarks_binary_list (list of int): Binary list indicating which landmarks are detected in the image.
    - orientation_radians (float): The robot's orientation in radians from the IMU.

    Returns:
    - full_features (numpy array): The combined feature vector from ResNet50, landmarks, and orientation.
    """
    with torch.no_grad():
        # Extract ResNet50 features
        features = resnet50(image_tensor.unsqueeze(0))  # Add batch dimension (unsqueeze adds a batch dimension)
        flattened_features = features.view(features.size(0), -1)  # Flatten to [1, 2048 * 7 * 7]
        flattened_features = flattened_features.cpu().numpy().squeeze()  # Convert to NumPy array and remove extra dimensions

    # Convert the binary landmark list and orientation to a NumPy array
    landmarks_and_orientation = np.array(landmarks_binary_list + [orientation_radians], dtype=np.float32)

    # Concatenate the ResNet features with the landmarks and orientation
    full_features = np.concatenate((flattened_features, landmarks_and_orientation))

    return full_features


# Function to extract features from each datapoint, including landmarks and orientation
def extract_features_with_landmarks_and_orientation(dataset):
    """
    Extract features from the dataset one by one, augmenting with landmarks and orientation.

    Args:
    - dataset (PovDataset): The dataset containing Datapoint objects.

    Returns:
    - features_list (list of numpy arrays): A list of feature vectors for each image in the dataset.
    - xy_labels (list of tuples): A list of (x, y) coordinates corresponding to each feature vector.
    """
    features_list = []
    xy_labels = []
    count = 1
    # Process each datapoint individually
    for datapoint in dataset.data:
        print("Processing data point: ", count)
        count += 1
        # Extract landmarks and orientation (simulated here as placeholders, replace with actual values)
        landmarks_binary_list = datapoint.landmark_mask  # Assume this is part of your dataset
        orientation_radians = datapoint.theta  # Assume this is part of your dataset

        # Extract features from the image tensor, and include the landmarks and orientation
        features = process_datapoint(datapoint.image_tensor, landmarks_binary_list, orientation_radians)
        features_list.append(features)

        # Keep track of the (x, y) coordinates
        xy_labels.append((datapoint.x, datapoint.y))

    return features_list, xy_labels

features, xy_list = extract_features_with_landmarks_and_orientation(data)


# Function to perform KMeans clustering
def cluster_with_kmeans(features_list, n_clusters):
    """
    Perform KMeans clustering on the extracted features.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    """
    # Convert feature list to a NumPy array for KMeans
    features_array = np.array(features_list)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Set random_state for reproducibility
    cluster_labels = kmeans.fit_predict(features_array)

    return cluster_labels

cluster_labels = cluster_with_kmeans(features, n_clusters)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/knn.png")

plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/knn_clusters.png", n_clusters=n_clusters)


def cluster_with_dbscan(features_list, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the extracted features.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint. Points classified as noise are labeled as -1.
    """
    # Convert feature list to a NumPy array for DBSCAN
    features_array = np.array(features_list)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_array)

    return cluster_labels

cluster_labels = cluster_with_dbscan(features)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/dbscan.png")

plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/dbscan_clusters.png", n_clusters=8)


def cluster_with_agglomerative(features_list, n_clusters=8):
    """
    Perform Agglomerative Clustering on the extracted features.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to find.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    """
    # Convert feature list to a NumPy array for Agglomerative Clustering
    features_array = np.array(features_list)

    # Perform Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = agglomerative.fit_predict(features_array)

    return cluster_labels


cluster_labels = cluster_with_agglomerative(features, n_clusters)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/agglomerative.png")

plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/agglomerative_clusters.png", n_clusters=n_clusters)


def cluster_with_spectral(features_list, n_clusters=8):
    """
    Perform Spectral Clustering on the extracted features.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to find.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    """
    # Convert feature list to a NumPy array for Spectral Clustering
    features_array = np.array(features_list)

    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    cluster_labels = spectral.fit_predict(features_array)

    return cluster_labels


cluster_labels = cluster_with_spectral(features, n_clusters)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/spectral.png")

plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/spectral_clusters.png", n_clusters=n_clusters)



def cluster_with_birch(features_list, n_clusters=8):
    """
    Perform BIRCH clustering on the extracted features.

    Args:
    - features_list (list of numpy arrays): The feature vectors extracted from the images.
    - n_clusters (int): The number of clusters to find.

    Returns:
    - cluster_labels (list of int): The cluster label for each datapoint.
    """
    # Convert feature list to a NumPy array for BIRCH
    features_array = np.array(features_list)

    # Perform BIRCH clustering
    birch = Birch(n_clusters=n_clusters)
    cluster_labels = birch.fit_predict(features_array)

    return cluster_labels


cluster_labels = cluster_with_birch(features, n_clusters)

# Now plot the clusters
plot_clusters(xy_list, cluster_labels, "data/figures/Clustering/birch.png")

plot_clusters_by_subplots(xy_list, cluster_labels, "data/figures/Clustering/birch_clusters.png", n_clusters=n_clusters)
