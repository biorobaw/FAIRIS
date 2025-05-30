import os
os.chdir("../../..")
print(os.getcwd())
import numpy as np
import json
from itertools import combinations

# Define landmarks with positions
landmarks = [
    (1.85, 1.85, (1.0, 0.0, 0.0)),
    (0.00, 2.61, (0.0, 1.0, 0.0)),
    (-1.85, 1.85, (0.0, 0.0, 1.0)),
    (-2.61, 0.00, (1.0, 1.0, 0.0)),
    (-1.85, -1.85, (0.0, 1.0, 1.0)),
    (0.00, -2.61, (1.0, 0.5, 0.0)),
    (1.85, -1.85, (0.5, 0.0, 0.5)),
    (2.61, 0.00, (0.5, 0.5, 0.0)),
]

# Generate filtered landmark pairs (excluding neighbors)
filtered_landmark_pairs = []
num_landmarks = len(landmarks)

for i, (x1, y1, _) in enumerate(landmarks):
    for j, (x2, y2, _) in enumerate(landmarks):
        if abs(i - j) > 1 and abs(i - j) < (num_landmarks - 1):  # Skip neighbors and wrap-around
            filtered_landmark_pairs.append(((x1, y1), (x2, y2)))

# Function to calculate (x, y) points along a line with margin from walls
def adjust_points_for_wall_margin(x1, y1, x2, y2, margin, num_points):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    dx_margin, dy_margin = (dx / length) * margin, (dy / length) * margin
    x1_margin, y1_margin = x1 + dx_margin, y1 + dy_margin
    x2_margin, y2_margin = x2 - dx_margin, y2 - dy_margin
    points = np.linspace([x1_margin, y1_margin], [x2_margin, y2_margin], num_points)
    return points

# Parameters
num_points = 20
wall_margin = 1.0  # 1m margin from walls
interval = np.pi / 4  # π/4 intervals for 8 thetas
thetas = [i * interval for i in range(8)]  # [0, π/4, ..., 7π/4]

# Generate (x, y, theta) for filtered pairs
expanded_pose_list = []

for (x1, y1), (x2, y2) in filtered_landmark_pairs:
    adjusted_points = adjust_points_for_wall_margin(x1, y1, x2, y2, wall_margin, num_points)
    for x, y in adjusted_points:
        for theta in thetas:
            expanded_pose_list.append((x, y, theta))

# Save the updated pose list to a JSON file
file_path = "data/PathPoints/VPC2.json"
with open(file_path, "w") as f:
    json.dump(expanded_pose_list, f)

print(f"Expanded pose list saved to {file_path}")
