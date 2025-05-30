import pickle
import os
import gc
import numpy as np
from sklearn.cluster import KMeans, Birch
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import math
from sklearn.metrics import homogeneity_score
from scipy.stats import entropy
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib.animation import FuncAnimation
from scipy.stats import kruskal
from scipy.stats import f_oneway, kruskal
from fairis_tools.experiment_tools.place_cell.PlaceCellLibrary import *

os.chdir("../../..")
print(os.getcwd())

with open("data/VisualPlaceCellData/LM8_Testing", 'rb') as file:
    test_data = pickle.load(file)

landmarks = [
    (1.85, 1.85, (1.00, 0.00, 0.00)),
    (0.00, 2.61, (0.00, 1.00, 0.00)),
    (-1.85, 1.85, (0.00, 0.00, 1.00)),
    (-2.61, 0.00, (1.00, 1.00, 0.00)),
    (-2.61, 0.00, (1.00, 0.00, 1.00)),
    (-1.85, -1.85, (0.00, 1.00, 1.00)),
    (0.00, -2.61, (1.00, 0.50, 0.00)),
    (1.85, -1.85, (0.50, 0.00, 0.50)),
    (2.61, 0.00, (0.50, 0.50, 0.00))
]

def group_dataset_by_theta(dataset):
    """
    Groups observations in the dataset by unique theta values. This function
    does not consider spatial proximity (x, y coordinates) during grouping.

    Parameters:
    - dataset: An object with an 'observations' attribute, which is a list of observations.
               Each observation is expected to have a 'theta' attribute.

    Returns:
    - dict: A dictionary where the keys are unique theta values, and the values are
            lists of observations corresponding to each theta.
    """
    # Initialize a dictionary to store the grouped observations by theta
    theta_groups = defaultdict(list)

    # Iterate through the observations and group them by their theta value
    for observation in dataset.observations:
        theta_groups[observation.theta].append(observation)

    return theta_groups


def select_knn_with_orientation(grouped_data, orientation, k, mode='nearest_same_orientation'):
    """
    Selects k data points based on spatial proximity and orientation.

    Parameters:
    - grouped_data: dict, output of group_dataset_by_theta function {theta: [data_points]}.
    - orientation: float, the orientation (theta) to select the initial data point from.
    - k: int, the number of neighbors to select.
    - mode: str, selection mode ('nearest_same_orientation', 'farthest_same_orientation',
             'farthest_different_orientation', 'nearest_different_orientation').

    Returns:
    - list: A list containing k selected data points based on the chosen mode.
    """
    if mode not in ['nearest_same_orientation', 'farthest_same_orientation',
                    'farthest_different_orientation', 'nearest_different_orientation']:
        raise ValueError("Invalid mode. Choose from 'nearest_same_orientation', 'farthest_same_orientation', "
                         "'farthest_different_orientation', or 'nearest_different_orientation'.")

    if mode == 'nearest_same_orientation':
        # Get all points with the same orientation
        if orientation not in grouped_data or len(grouped_data[orientation]) == 0:
            raise ValueError(f"No data points available for the specified orientation: {orientation}")

        data_points = grouped_data[orientation]

        # Randomly select an initial data point
        selected_point = random.choice(data_points)
        selected_point_coords = np.array([selected_point.x, selected_point.y])

        # Find the k nearest neighbors
        distances = cdist([selected_point_coords], [(dp.x, dp.y) for dp in data_points])[0]
        nearest_indices = np.argsort(distances)[:k + 1]  # +1 to include the selected point

        # Return the initial point and k nearest neighbors
        return [data_points[i] for i in nearest_indices if data_points[i] != selected_point][:k]

    if mode == 'nearest_different_orientation':
        # Get all points with different orientations
        candidate_points = [dp for theta, points in grouped_data.items() if theta != orientation for dp in points]
        if len(candidate_points) == 0:
            raise ValueError("No data points available with different orientations.")

        # Randomly select an initial data point
        selected_point = random.choice(candidate_points)
        selected_point_coords = np.array([selected_point.x, selected_point.y])

        # Compute distances to all candidates
        distances = cdist([selected_point_coords], [(dp.x, dp.y) for dp in candidate_points])[0]
        nearest_indices = np.argsort(distances)[:k]  # Select k nearest neighbors

        return [candidate_points[i] for i in nearest_indices]

    if mode in ['farthest_same_orientation', 'farthest_different_orientation']:
        if mode == 'farthest_same_orientation':
            if orientation not in grouped_data or len(grouped_data[orientation]) == 0:
                raise ValueError(f"No data points available for the specified orientation: {orientation}")
            candidate_points = grouped_data[orientation]
        else:
            # Different orientation points
            candidate_points = [dp for theta, points in grouped_data.items() if theta != orientation for dp in points]
            if len(candidate_points) == 0:
                raise ValueError("No data points available with different orientations.")

        # Step 3: Convert points to (x, y) coordinates
        candidate_coords = np.array([(dp.x, dp.y) for dp in candidate_points])

        # Step 4: Greedy selection of k points that maximize pairwise distances
        # Randomly select the first point
        selected_indices = [random.randint(0, len(candidate_coords) - 1)]
        selected_points = [candidate_points[selected_indices[0]]]

        while len(selected_indices) < k:
            # Calculate distances from the current set of selected points to all candidates
            distances_to_selected = np.min(cdist(candidate_coords, candidate_coords[selected_indices]), axis=1)

            # Exclude already selected points
            distances_to_selected[selected_indices] = -np.inf

            # Select the point with the maximum distance to the current selection
            next_index = np.argmax(distances_to_selected)
            selected_indices.append(next_index)
            selected_points.append(candidate_points[next_index])

        return selected_points

def calculate_similarity_metrics(grouped_data, pc_network, n=20, mode='nearest_same_orientation',
                                 feature_type='multimodal'):
    """
    Calculate similarity metrics by repeatedly grouping data points using the specified mode.

    Parameters:
    - grouped_data: dict, output of group_dataset_by_theta function {theta: [data_points]}.
    - pc_network: object, the place cell network used to get activations.
    - n: int, number of times to perform grouping and calculate similarity metrics (default 20).
    - mode: str, selection mode ('nearest_same_orientation', 'farthest_same_orientation', or 'farthest_different_orientation').
    - feature_type: str, feature type used to calculate activations ('cnn' or 'multimodal').

    Returns:
    - DataFrame: A dataset of cosine similarity, Pearson correlation, and Euclidean distance for each iteration.
    - dict: The average values of each similarity metric across all iterations.
    """
    results = []

    for i in range(n):
        # Step 1: Select a group of data points using the specified mode
        selected_points = select_knn_with_orientation(grouped_data,
                                                      orientation=random.choice(list(grouped_data.keys())), k=5,
                                                      mode=mode)

        # Step 2: Compute place cell activations for each selected point
        activation_vectors = []
        for dp in selected_points:
            if feature_type == 'cnn':
                activations = pc_network.get_all_pc_activations_normalized(dp.cnn_feature_vector, norm_type='min_max')
            else:
                activations = pc_network.get_all_pc_activations_normalized(dp.multimodal_feature_vector,
                                                                           norm_type='min_max')
            activation_vectors.append(activations)

        activation_vectors = np.array(activation_vectors)

        # Step 3: Calculate similarity metrics
        cosine_sim_matrix = 1 - squareform(pdist(activation_vectors, metric='cosine'))
        euclidean_dist_matrix = squareform(pdist(activation_vectors, metric='euclidean'))
        n_place_cells = len(pc_network.pc_list)
        euclidean_scale = np.sqrt(n_place_cells)

        # Pearson correlation matrix
        num_vectors = len(activation_vectors)
        pearson_corr_matrix = np.zeros((num_vectors, num_vectors))
        for j in range(num_vectors):
            for k in range(num_vectors):
                if j != k:
                    pearson_corr, _ = pearsonr(activation_vectors[j], activation_vectors[k])
                    pearson_corr_matrix[j, k] = pearson_corr
                else:
                    pearson_corr_matrix[j, k] = 1  # Correlation with itself

        # Step 4: Calculate average metrics for the current iteration
        avg_cosine_sim = np.mean(cosine_sim_matrix[np.triu_indices(num_vectors, k=1)])
        avg_pearson_corr = np.mean(pearson_corr_matrix[np.triu_indices(num_vectors, k=1)])
        avg_euclidean_dist = np.mean(euclidean_dist_matrix[np.triu_indices(num_vectors, k=1)])

        # Store the results
        results.append({
            'Iteration': i + 1,
            'Avg Cosine Similarity': avg_cosine_sim,
            'Avg Pearson Correlation': avg_pearson_corr,
            'Avg Euclidean Distance': avg_euclidean_dist
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Calculate overall averages
    averages = {
        'Average Cosine Similarity': results_df['Avg Cosine Similarity'].mean(),
        'Average Pearson Correlation': results_df['Avg Pearson Correlation'].mean(),
        'Average Euclidean Distance': results_df['Avg Euclidean Distance'].mean()
    }

    return results_df, averages


def analyze_similarity_across_modes(grouped_data, pc_network, n=20, feature_type='multimodal'):
    """
    Perform similarity analysis across different grouping modes, calculate averages,
    and test for significant differences between the modes using raw samples.

    Parameters:
    - grouped_data: dict, output of group_dataset_by_theta function {theta: [data_points]}.
    - pc_network: object, the place cell network used to get activations.
    - n: int, number of iterations for each mode (default 20).
    - feature_type: str, feature type used to calculate activations ('cnn' or 'multimodal').

    Returns:
    - None, but prints the summary of averages and statistical test findings.
    """

    # Modes to analyze
    modes = ['nearest_same_orientation', 'farthest_same_orientation',
             'farthest_different_orientation', 'nearest_different_orientation']

    # Store raw DataFrames for statistical testing
    combined_df_list = []

    for mode in modes:
        # Calculate similarity metrics for the given mode
        results_df, averages = calculate_similarity_metrics(grouped_data, pc_network, n=n, mode=mode,
                                                            feature_type=feature_type)

        # Add the mode as a column to the results DataFrame
        results_df['Mode'] = mode
        combined_df_list.append(results_df)

        # Print averages for this mode
        print(f"\nAverages for {mode}:")
        for metric, avg in averages.items():
            print(f"  {metric}: {avg:.4f}")

    # Combine all the results into a single DataFrame
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    # Step 2: Perform ANOVA using raw results
    print("\n\n=== ANOVA Tests ===")
    metrics = ['Avg Cosine Similarity', 'Avg Pearson Correlation', 'Avg Euclidean Distance']

    for metric in metrics:
        # Perform ANOVA on the raw results
        f_stat, p_value = f_oneway(combined_df[combined_df['Mode'] == 'nearest_same_orientation'][metric],
                                   combined_df[combined_df['Mode'] == 'nearest_different_orientation'][metric],
                                   combined_df[combined_df['Mode'] == 'farthest_same_orientation'][metric],
                                   combined_df[combined_df['Mode'] == 'farthest_different_orientation'][metric])
        print(f"\nANOVA for {metric}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Significant difference found in {metric} across modes (p < 0.05)")
        else:
            print(f"  No significant difference in {metric} (p = {p_value:.4f})")

    # Step 3: Perform Kruskal-Wallis test as a backup
    print("\n=== Backup Kruskal-Wallis Tests ===")
    for metric in metrics:
        stat, p_value = kruskal(combined_df[combined_df['Mode'] == 'nearest_same_orientation'][metric],
                                combined_df[combined_df['Mode'] == 'nearest_different_orientation'][metric],
                                combined_df[combined_df['Mode'] == 'farthest_same_orientation'][metric],
                                combined_df[combined_df['Mode'] == 'farthest_different_orientation'][metric])
        print(f"Kruskal-Wallis test for {metric}: Statistic = {stat:.4f}, p-value = {p_value:.4f}")

feature_modes = ['multimodal', 'cnn']
num_pc = [10,20,100,500]
for feature_mode in feature_modes:
    for n in num_pc:
        file_name = feature_mode+'_kmeans_'+str(n)+'centers'
        with open("data/VisualPlaceCellData/VisualPlaceCellClusters/multimodal_kmeans_20centers",'rb') as file:
            data = pickle.load(file)

        pc_network = VisualPlaceCellNetwork()
        for cluster in data:
            pc_network.add_pc_to_network(cluster[0],radius=cluster[1])

        del data
        out1 = gc.collect()

        # Group the data by Theta
        grouped_data = group_dataset_by_theta(test_data)
        analyze_similarity_across_modes(grouped_data, pc_network, n=n, feature_type=feature_mode)
